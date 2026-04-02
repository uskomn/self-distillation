# pretrain_distill.py
"""
第一阶段：任务感知预训练蒸馏（Task-Aware Pretraining Distillation）

损失组成：
  Wikipedia batch：L_MLM + L_hidden + L_attn        （通用语言表征）
  QA batch        ：L_QA_hard + L_QA_kd             （任务意识注入）

两种 batch 交替出现，比例由 PT_QA_INTERLEAVE_EVERY 控制：
  每 PT_QA_INTERLEAVE_EVERY 个 Wiki batch 插入 1 个 QA batch

教师模型：
  - Wikipedia 阶段：hfl/chinese-roberta-wwm-ext 预训练权重（提供隐层/attention）
  - QA 阶段：      CMRC2018 微调后教师（提供 QA logits）
"""

import os
import bz2
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk, Dataset,load_dataset
from tqdm import tqdm

try:
    import mwxml
    HAS_MWXML = True
except ImportError:
    HAS_MWXML = False

from config import (
    TEACHER_NAME, TEACHER_SAVE_PATH,
    STUDENT_NUM_LAYERS, TEACHER_LAYERS_FOR_DISTILL,
    PRETRAIN_STUDENT_SAVE_PATH,
    PRETRAIN_DATA_SOURCE,
    PRETRAIN_LOCAL_XML_PATH, PRETRAIN_LOCAL_CACHE_DIR,
    PRETRAIN_WIKI_NAME, PRETRAIN_WIKI_CONFIG,
    PRETRAIN_CC100_NAME, PRETRAIN_CC100_CONFIG,
    PRETRAIN_MAX_SAMPLES,
    PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE, PRETRAIN_LR,
    PRETRAIN_WARMUP_RATIO, PRETRAIN_WEIGHT_DECAY, PRETRAIN_MLM_PROB,
    MAX_LENGTH, DOC_STRIDE,
    PT_LAMBDA_MLM, PT_LAMBDA_HIDDEN, PT_LAMBDA_ATTN,
    PT_QA_INTERLEAVE_EVERY,
    PT_LAMBDA_QA_HARD, PT_LAMBDA_QA_KD,
    PT_USE_QA_KD,
    TEMPERATURE,
)


# ============================================================
# 教师模型
# ============================================================

class PretrainTeacher(nn.Module):
    """
    预训练阶段教师（提供隐层/attention 蒸馏信号）
    使用原始预训练权重，冻结
    """
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(TEACHER_NAME)
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(TEACHER_NAME, config=config)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return outputs.hidden_states, outputs.attentions


class QATeacher(nn.Module):
    """
    QA 阶段教师（提供 QA logits 蒸馏信号）
    加载 CMRC2018 微调后的教师 checkpoint，冻结
    """
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(TEACHER_SAVE_PATH)
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            TEACHER_SAVE_PATH, config=config
        )
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return outputs.start_logits, outputs.end_logits


# ============================================================
# 学生模型（预训练阶段带 MLM head）
# ============================================================

class PretrainStudent(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained(TEACHER_NAME)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = BertForMaskedLM(config)
        self._init_from_teacher()

    def _init_from_teacher(self):
        print(f"从教师预训练权重初始化学生（层对齐：{TEACHER_LAYERS_FOR_DISTILL}）...")
        teacher_model = AutoModel.from_pretrained(TEACHER_NAME)
        self.model.bert.embeddings.load_state_dict(
            teacher_model.embeddings.state_dict()
        )
        for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
            self.model.bert.encoder.layer[student_idx].load_state_dict(
                teacher_model.encoder.layer[teacher_idx].state_dict()
            )
        del teacher_model
        torch.cuda.empty_cache()
        print("学生初始化完成。")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )

    def get_bert_output(self, input_ids, attention_mask, token_type_ids=None):
        """
        QA batch 时只需要 bert encoder 的输出（hidden states），
        不做 MLM，直接取 bert 子模块输出
        """
        return self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )


# ============================================================
# QA head（用于任务感知阶段的 span 预测）
# ============================================================

class QAHead(nn.Module):
    """
    轻量 QA head，接在 PretrainStudent 的 bert encoder 后，
    预训练阶段用于接收 QA 任务监督信号。
    微调阶段会被 BertForQuestionAnswering 的 qa_outputs 替代。
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


# ============================================================
# 损失函数
# ============================================================

def wiki_distill_loss(student_outputs, teacher_hidden, teacher_attn,
                      lambda_mlm=PT_LAMBDA_MLM,
                      lambda_hidden=PT_LAMBDA_HIDDEN,
                      lambda_attn=PT_LAMBDA_ATTN):
    """Wikipedia batch 损失：MLM + 隐层对齐 + Attention 对齐"""
    mse = nn.MSELoss()

    l_mlm = student_outputs.loss
    if l_mlm is None:
        raise ValueError("MLM loss 为 None，请确认传入了 labels")
    if student_outputs.hidden_states is None or student_outputs.attentions is None:
        raise ValueError("hidden_states 或 attentions 为 None，请确认 forward 传入了对应参数")

    l_hidden = torch.tensor(0.0, device=l_mlm.device)
    for s_idx, t_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_h = student_outputs.hidden_states[s_idx + 1]
        t_h = teacher_hidden[t_idx + 1].detach()
        l_hidden = l_hidden + mse(F.normalize(s_h, dim=-1), F.normalize(t_h, dim=-1))
    l_hidden = l_hidden / len(TEACHER_LAYERS_FOR_DISTILL)

    l_attn = torch.tensor(0.0, device=l_mlm.device)
    for s_idx, t_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_a = student_outputs.attentions[s_idx]
        t_a = teacher_attn[t_idx].detach()
        l_attn = l_attn + mse(torch.sqrt(s_a + 1e-6), torch.sqrt(t_a + 1e-6))
    l_attn = l_attn / len(TEACHER_LAYERS_FOR_DISTILL)

    total = lambda_mlm * l_mlm + lambda_hidden * l_hidden + lambda_attn * l_attn
    return total, l_mlm, l_hidden, l_attn


def qa_task_loss(student_start, student_end,
                 start_positions, end_positions,
                 teacher_start=None, teacher_end=None,
                 lambda_hard=PT_LAMBDA_QA_HARD,
                 lambda_kd=PT_LAMBDA_QA_KD,
                 temperature=TEMPERATURE):
    """
    QA batch 损失：hard label CE + 教师 logits KD
    teacher_start/end 为 None 时只用 hard label
    """
    ce = nn.CrossEntropyLoss()
    l_hard = ce(student_start, start_positions) + ce(student_end, end_positions)

    if teacher_start is not None and teacher_end is not None:
        T = temperature
        l_kd = (
            F.kl_div(F.log_softmax(student_start / T, dim=-1),
                     F.softmax(teacher_start.detach() / T, dim=-1),
                     reduction="batchmean") +
            F.kl_div(F.log_softmax(student_end / T, dim=-1),
                     F.softmax(teacher_end.detach() / T, dim=-1),
                     reduction="batchmean")
        ) * (T ** 2)
        total = lambda_hard * l_hard + lambda_kd * l_kd
        return total, l_hard, l_kd
    else:
        return lambda_hard * l_hard, l_hard, torch.tensor(0.0)


# ============================================================
# 数据加载：Wikipedia
# ============================================================

def _clean_wikitext(raw):
    if raw.strip().lower().startswith("#redirect"):
        return ""
    text = re.sub(r"\{\{[^}]*\}\}", "", raw)
    text = re.sub(r"\[\[(File|Image|文件|图像|图片):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"=+([^=]+)=+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_local_xml():
    cache_file = os.path.join(PRETRAIN_LOCAL_CACHE_DIR, "texts.jsonl")
    if os.path.exists(cache_file):
        print(f"发现解析缓存，直接加载：{cache_file}")
        texts = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="读取缓存"):
                texts.append(json.loads(line)["text"])
                if PRETRAIN_MAX_SAMPLES and len(texts) >= PRETRAIN_MAX_SAMPLES:
                    break
        print(f"缓存加载完成，共 {len(texts)} 条")
        return texts

    os.makedirs(PRETRAIN_LOCAL_CACHE_DIR, exist_ok=True)
    print(f"开始解析：{PRETRAIN_LOCAL_XML_PATH}")
    texts = []

    if HAS_MWXML:
        print("使用 mwxml 解析")
        dump = mwxml.Dump.from_file(bz2.open(PRETRAIN_LOCAL_XML_PATH, "rb"))
        with open(cache_file, "w", encoding="utf-8") as cache_f:
            for page in tqdm(dump.pages, desc="解析页面"):
                if page.namespace != 0:
                    continue
                for revision in page:
                    raw = revision.text or ""
                    text = _clean_wikitext(raw)
                    if len(text) >= 50:
                        texts.append(text)
                        cache_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    break
                if PRETRAIN_MAX_SAMPLES and len(texts) >= PRETRAIN_MAX_SAMPLES:
                    break
    else:
        print("mwxml 未安装，使用正则解析（可 pip install mwxml 加速）")
        with open(cache_file, "w", encoding="utf-8") as cache_f:
            with bz2.open(PRETRAIN_LOCAL_XML_PATH, "rt", encoding="utf-8") as xml_f:
                in_text, buf = False, []
                for line in tqdm(xml_f, desc="解析 XML 行"):
                    if "<text" in line:
                        m = re.search(r"<text[^>]*>(.*?)</text>", line, re.DOTALL)
                        if m:
                            text = _clean_wikitext(m.group(1))
                            if len(text) >= 50:
                                texts.append(text)
                                cache_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        else:
                            m2 = re.search(r"<text[^>]*>(.*)", line, re.DOTALL)
                            buf = [m2.group(1)] if m2 else []
                            in_text = True
                    elif in_text:
                        if "</text>" in line:
                            buf.append(line[:line.index("</text>")])
                            text = _clean_wikitext("".join(buf))
                            if len(text) >= 50:
                                texts.append(text)
                                cache_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                            in_text, buf = False, []
                        else:
                            buf.append(line)
                    if PRETRAIN_MAX_SAMPLES and len(texts) >= PRETRAIN_MAX_SAMPLES:
                        break

    print(f"解析完成，共 {len(texts)} 条")
    return texts


def _load_raw_dataset():
    source = PRETRAIN_DATA_SOURCE.lower()
    split = "train[:{}]".format(PRETRAIN_MAX_SAMPLES)
    if source == "local_xml":
        texts = _parse_local_xml()
        return Dataset.from_dict({"text": texts[:PRETRAIN_MAX_SAMPLES] if PRETRAIN_MAX_SAMPLES else texts})
    elif source == "wikimedia":
        ds = load_dataset(PRETRAIN_WIKI_NAME, PRETRAIN_WIKI_CONFIG, split=split, trust_remote_code=True)
        return ds.select_columns(["text"])
    elif source == "cc100":
        ds = load_dataset(PRETRAIN_CC100_NAME, PRETRAIN_CC100_CONFIG, split=split, trust_remote_code=True)
        return ds.select_columns(["text"])
    else:
        raise ValueError(f"未知数据源：{source}")


def get_wiki_dataloader(tokenizer):
    print(f"加载 Wikipedia 数据，来源：{PRETRAIN_DATA_SOURCE}，上限：{PRETRAIN_MAX_SAMPLES}")
    dataset = _load_raw_dataset()

    def tokenize_fn(examples):
        texts = [t if (t and t.strip()) else "空" for t in examples["text"]]
        return tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=dataset.column_names, desc="Tokenizing Wiki")
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=PRETRAIN_MLM_PROB
    )
    loader = DataLoader(tokenized, batch_size=PRETRAIN_BATCH_SIZE,
                        shuffle=True, collate_fn=collator)
    print(f"Wikipedia 数据就绪：{len(tokenized)} 条，{len(loader)} 个 batch")
    return loader


# ============================================================
# 数据加载：CMRC2018 QA
# ============================================================

def _preprocess_qa(examples, tokenizer):
    """CMRC2018 预处理，复用 experiments.py 里的逻辑"""
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]
        sequence_ids = tokenized.sequence_ids(i)

        ctx_start = next((j for j, s in enumerate(sequence_ids) if s == 1), None)
        ctx_end = next((j for j, s in reversed(list(enumerate(sequence_ids))) if s == 1), None)

        ans_start_char = answers["answer_start"][0]
        ans_text = answers["text"][0]
        ans_end_char = ans_start_char + len(ans_text)

        if (ctx_start is None or ctx_end is None or
                offsets[ctx_start][0] > ans_end_char or
                offsets[ctx_end][1] < ans_start_char):
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            tok_s = ctx_start
            while tok_s <= ctx_end and offsets[tok_s][0] <= ans_start_char:
                tok_s += 1
            tok_e = ctx_end
            while tok_e >= ctx_start and offsets[tok_e][1] >= ans_end_char:
                tok_e -= 1
            tokenized["start_positions"].append(tok_s - 1)
            tokenized["end_positions"].append(tok_e + 1)

    return tokenized


def get_qa_dataloader(tokenizer):
    """加载 CMRC2018 训练集，用于任务感知阶段"""
    print("加载 CMRC2018 QA 数据（任务感知用）...")
    dataset = load_from_disk("/root/autodl-tmp/cmrc2018")["train"]
    tokenized = dataset.map(
        lambda x: _preprocess_qa(x, tokenizer),
        batched=True, remove_columns=dataset.column_names,
        desc="Tokenizing CMRC2018",
    )
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=PRETRAIN_BATCH_SIZE,
                        shuffle=True, drop_last=True)
    print(f"CMRC2018 QA 数据就绪：{len(tokenized)} 条，{len(loader)} 个 batch")
    return loader


# ============================================================
# 主训练流程
# ============================================================

def run_pretrain_distill():
    print("=" * 60)
    print("第一阶段：任务感知预训练蒸馏")
    print(f"  Wiki batch 每 {PT_QA_INTERLEAVE_EVERY} 个插入 1 个 QA batch")
    print(f"  QA 损失：hard={PT_LAMBDA_QA_HARD}, kd={PT_LAMBDA_QA_KD if PT_USE_QA_KD else 'off'}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)

    # 教师
    pretrain_teacher = PretrainTeacher().to(device)
    pretrain_teacher.eval()

    qa_teacher = None
    if PT_USE_QA_KD and os.path.exists(TEACHER_SAVE_PATH):
        print(f"加载 QA 教师：{TEACHER_SAVE_PATH}")
        qa_teacher = QATeacher().to(device)
        qa_teacher.eval()
    else:
        print("QA 教师不可用，QA 阶段只使用 hard label 损失")

    # 学生
    student_wrapper = PretrainStudent()
    student = student_wrapper.model.to(device)

    # QA head（附加在学生 bert 后，只在预训练阶段使用）
    hidden_size = student.config.hidden_size
    qa_head = QAHead(hidden_size).to(device)

    # 数据
    wiki_loader = get_wiki_dataloader(tokenizer)
    qa_loader = get_qa_dataloader(tokenizer)
    qa_iter = cycle(qa_loader)  # QA 数据循环使用（量少）

    # 优化器（学生 bert + qa_head 一起优化）
    all_params = list(student.parameters()) + list(qa_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=PRETRAIN_LR, weight_decay=PRETRAIN_WEIGHT_DECAY)
    total_steps = len(wiki_loader) * PRETRAIN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * PRETRAIN_WARMUP_RATIO), total_steps
    )

    best_loss = float("inf")

    for epoch in range(PRETRAIN_EPOCHS):
        student.train()
        qa_head.train()

        # 统计各损失
        stats = {k: 0.0 for k in ["total", "mlm", "hidden", "attn",
                                    "qa_hard", "qa_kd", "wiki_steps", "qa_steps"]}

        pbar = tqdm(wiki_loader, desc=f"[任务感知预训练蒸馏 Epoch {epoch+1}/{PRETRAIN_EPOCHS}]")

        for wiki_step, wiki_batch in enumerate(pbar):
            # ========== Wikipedia batch ==========
            input_ids      = wiki_batch["input_ids"].to(device)
            attention_mask = wiki_batch["attention_mask"].to(device)
            labels         = wiki_batch["labels"].to(device)
            token_type_ids = wiki_batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            t_hidden, t_attn = pretrain_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            s_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss_wiki, l_mlm, l_hidden, l_attn = wiki_distill_loss(s_out, t_hidden, t_attn)

            loss_wiki.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            stats["total"]      += loss_wiki.item()
            stats["mlm"]        += l_mlm.item()
            stats["hidden"]     += l_hidden.item()
            stats["attn"]       += l_attn.item()
            stats["wiki_steps"] += 1

            # ========== QA batch（每 N 步插入一次）==========
            if (wiki_step + 1) % PT_QA_INTERLEAVE_EVERY == 0:
                qa_batch = next(qa_iter)
                qa_input_ids      = qa_batch["input_ids"].to(device)
                qa_attention_mask = qa_batch["attention_mask"].to(device)
                qa_token_type_ids = qa_batch["token_type_ids"].to(device)
                start_pos         = qa_batch["start_positions"].to(device)
                end_pos           = qa_batch["end_positions"].to(device)

                # 学生 bert encoder 输出
                bert_out = student_wrapper.get_bert_output(
                    input_ids=qa_input_ids,
                    attention_mask=qa_attention_mask,
                    token_type_ids=qa_token_type_ids,
                )
                s_start, s_end = qa_head(bert_out.last_hidden_state)

                # 教师 QA logits
                t_start = t_end = None
                if qa_teacher is not None:
                    t_start, t_end = qa_teacher(
                        input_ids=qa_input_ids,
                        attention_mask=qa_attention_mask,
                        token_type_ids=qa_token_type_ids,
                    )

                loss_qa, l_qa_hard, l_qa_kd = qa_task_loss(
                    s_start, s_end, start_pos, end_pos, t_start, t_end
                )

                loss_qa.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                stats["qa_hard"]  += l_qa_hard.item()
                stats["qa_kd"]    += l_qa_kd.item() if isinstance(l_qa_kd, torch.Tensor) else l_qa_kd
                stats["qa_steps"] += 1

            # 更新进度条
            pbar.set_postfix(
                wiki=f"{loss_wiki.item():.3f}",
                mlm=f"{l_mlm.item():.3f}",
                qa_h=f"{stats['qa_hard']/max(stats['qa_steps'],1):.3f}",
            )

        # Epoch 汇总
        ws = stats["wiki_steps"]
        qs = max(stats["qa_steps"], 1)
        avg_total = stats["total"] / ws
        print(
            f"Epoch {epoch+1} | "
            f"Wiki Total:{avg_total:.4f} MLM:{stats['mlm']/ws:.4f} "
            f"Hidden:{stats['hidden']/ws:.4f} Attn:{stats['attn']/ws:.4f} | "
            f"QA Hard:{stats['qa_hard']/qs:.4f} QA KD:{stats['qa_kd']/qs:.4f}"
        )

        # 保存最优权重（只保存 bert 部分，qa_head 预训练后丢弃）
        if avg_total < best_loss:
            best_loss = avg_total
            os.makedirs(PRETRAIN_STUDENT_SAVE_PATH, exist_ok=True)
            student.save_pretrained(PRETRAIN_STUDENT_SAVE_PATH)
            tokenizer.save_pretrained(PRETRAIN_STUDENT_SAVE_PATH)
            print(f" 保存最优权重 Loss:{avg_total:.4f} -> {PRETRAIN_STUDENT_SAVE_PATH}")

    print(f"\n任务感知预训练蒸馏完成，最优 Loss: {best_loss:.4f}")
    print("注：qa_head 为预训练阶段临时模块，微调阶段将使用 BertForQuestionAnswering 的 qa_outputs 替代")


if __name__ == "__main__":
    run_pretrain_distill()