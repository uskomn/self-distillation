# pretrain_distill.py
"""
第一阶段：预训练蒸馏
教师：hfl/chinese-roberta-wwm-ext（原始预训练权重，冻结）
学生：6层 BERT，从教师偶数层初始化
数据：本地 Wikipedia XML bz2（或在线备选）
损失：MLM Loss + 隐层 MSE + Attention MSE

完成后学生权重保存至 PRETRAIN_STUDENT_SAVE_PATH，
供第二阶段（微调蒸馏）作为学生初始化权重使用。
"""

import os
import bz2
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, Dataset
from tqdm import tqdm

try:
    import mwxml
    HAS_MWXML = True
except ImportError:
    HAS_MWXML = False

from config import (
    TEACHER_NAME,
    STUDENT_NUM_LAYERS,
    TEACHER_LAYERS_FOR_DISTILL,
    PRETRAIN_STUDENT_SAVE_PATH,
    PRETRAIN_DATA_SOURCE,
    PRETRAIN_LOCAL_XML_PATH,
    PRETRAIN_LOCAL_CACHE_DIR,
    PRETRAIN_WIKI_NAME,
    PRETRAIN_WIKI_CONFIG,
    PRETRAIN_CC100_NAME,
    PRETRAIN_CC100_CONFIG,
    PRETRAIN_MAX_SAMPLES,
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_LR,
    PRETRAIN_WARMUP_RATIO,
    PRETRAIN_WEIGHT_DECAY,
    PRETRAIN_MLM_PROB,
    MAX_LENGTH,
    PT_LAMBDA_MLM,
    PT_LAMBDA_HIDDEN,
    PT_LAMBDA_ATTN,
)


# ============================================================
# 教师
# ============================================================

class PretrainTeacher(nn.Module):
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


# ============================================================
# 学生
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
        )


# ============================================================
# 预训练蒸馏损失
# ============================================================

def pretrain_distill_loss(
        student_outputs,
        teacher_hidden_states,
        teacher_attentions,
        lambda_mlm=PT_LAMBDA_MLM,
        lambda_hidden=PT_LAMBDA_HIDDEN,
        lambda_attn=PT_LAMBDA_ATTN,
):
    mse = nn.MSELoss()

    l_mlm = student_outputs.loss

    l_hidden = 0.0
    for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_h = student_outputs.hidden_states[student_idx + 1]
        t_h = teacher_hidden_states[teacher_idx + 1].detach()
        l_hidden += mse(F.normalize(s_h, dim=-1), F.normalize(t_h, dim=-1))
    l_hidden /= len(TEACHER_LAYERS_FOR_DISTILL)

    l_attn = 0.0
    for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_a = student_outputs.attentions[student_idx]
        t_a = teacher_attentions[teacher_idx].detach()
        l_attn += mse(torch.sqrt(s_a + 1e-6), torch.sqrt(t_a + 1e-6))
    l_attn /= len(TEACHER_LAYERS_FOR_DISTILL)

    total = lambda_mlm * l_mlm + lambda_hidden * l_hidden + lambda_attn * l_attn
    return total, l_mlm, l_hidden, l_attn


# ============================================================
# Wikipedia XML 本地解析
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
    """
    解析本地 zhwiki XML bz2，提取纯文本。
    首次运行解析并缓存到 PRETRAIN_LOCAL_CACHE_DIR/texts.jsonl，
    后续直接读缓存，无需重复解析。
    """
    cache_file = os.path.join(PRETRAIN_LOCAL_CACHE_DIR, "texts.jsonl")

    # 有缓存直接读
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
    print("首次解析耗时较长，结果将缓存至：" + cache_file)

    texts = []

    if HAS_MWXML:
        print("使用 mwxml 解析（推荐）")
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
                    break  # 每页只取最新版本
                if PRETRAIN_MAX_SAMPLES and len(texts) >= PRETRAIN_MAX_SAMPLES:
                    break
    else:
        print("mwxml 未安装，使用正则解析（可 pip install mwxml 加速）")
        with open(cache_file, "w", encoding="utf-8") as cache_f:
            with bz2.open(PRETRAIN_LOCAL_XML_PATH, "rt", encoding="utf-8") as xml_f:
                in_text = False
                buf = []
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
                            in_text = False
                            buf = []
                        else:
                            buf.append(line)
                    if PRETRAIN_MAX_SAMPLES and len(texts) >= PRETRAIN_MAX_SAMPLES:
                        break

    print(f"解析完成，共 {len(texts)} 条")
    return texts


# ============================================================
# 数据加载
# ============================================================

def _load_raw_dataset():
    source = PRETRAIN_DATA_SOURCE.lower()
    split = "train[:{}]".format(PRETRAIN_MAX_SAMPLES)

    if source == "local_xml":
        texts = _parse_local_xml()
        if PRETRAIN_MAX_SAMPLES:
            texts = texts[:PRETRAIN_MAX_SAMPLES]
        return Dataset.from_dict({"text": texts})

    elif source == "wikimedia":
        print(f"加载在线 Wikipedia：{PRETRAIN_WIKI_NAME} / {PRETRAIN_WIKI_CONFIG}")
        ds = load_dataset(PRETRAIN_WIKI_NAME, PRETRAIN_WIKI_CONFIG,
                          split=split, trust_remote_code=True)
        return ds.select_columns(["text"])

    elif source == "cc100":
        print(f"加载 CC-100：{PRETRAIN_CC100_NAME} / {PRETRAIN_CC100_CONFIG}")
        ds = load_dataset(PRETRAIN_CC100_NAME, PRETRAIN_CC100_CONFIG,
                          split=split, trust_remote_code=True)
        return ds.select_columns(["text"])

    else:
        raise ValueError(
            "未知数据源：{}，请在 config.py 将 PRETRAIN_DATA_SOURCE 设为 "
            "local_xml / wikimedia / cc100".format(source)
        )


def get_pretrain_dataloader(tokenizer):
    print(f"数据源：{PRETRAIN_DATA_SOURCE}，上限：{PRETRAIN_MAX_SAMPLES} 条")
    dataset = _load_raw_dataset()
    print(f"原始数据 {len(dataset)} 条")

    def tokenize_fn(examples):
        texts = [t if (t and t.strip()) else "空" for t in examples["text"]]
        return tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length")

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=PRETRAIN_MLM_PROB,
    )
    loader = DataLoader(tokenized, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator)
    print(f"数据就绪：{len(tokenized)} 条，{len(loader)} 个 batch")
    return loader


# ============================================================
# 主训练流程
# ============================================================

def run_pretrain_distill():
    print("=" * 60)
    print("第一阶段：预训练蒸馏")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    teacher = PretrainTeacher().to(device)
    teacher.eval()

    student_wrapper = PretrainStudent()
    student = student_wrapper.model.to(device)

    train_loader = get_pretrain_dataloader(tokenizer)

    optimizer = torch.optim.AdamW(student.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WEIGHT_DECAY)
    total_steps = len(train_loader) * PRETRAIN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * PRETRAIN_WARMUP_RATIO), total_steps
    )

    best_loss = float("inf")

    for epoch in range(PRETRAIN_EPOCHS):
        student.train()
        total_sum = mlm_sum = hidden_sum = attn_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[预训练蒸馏 Epoch {epoch+1}/{PRETRAIN_EPOCHS}]")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            t_hidden, t_attn = teacher(input_ids, attention_mask, token_type_ids)
            s_out = student(input_ids, attention_mask, token_type_ids, labels)

            loss, l_mlm, l_hidden, l_attn = pretrain_distill_loss(s_out, t_hidden, t_attn)

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_sum  += loss.item()
            mlm_sum    += l_mlm.item()
            hidden_sum += l_hidden.item()
            attn_sum   += l_attn.item()

            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                mlm=f"{l_mlm.item():.4f}",
                hid=f"{l_hidden.item():.4f}",
                attn=f"{l_attn.item():.4f}",
            )

        n = len(train_loader)
        avg = total_sum / n
        print(f"Epoch {epoch+1} | Total:{avg:.4f} MLM:{mlm_sum/n:.4f} "
              f"Hidden:{hidden_sum/n:.4f} Attn:{attn_sum/n:.4f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(PRETRAIN_STUDENT_SAVE_PATH, exist_ok=True)
            student.save_pretrained(PRETRAIN_STUDENT_SAVE_PATH)
            tokenizer.save_pretrained(PRETRAIN_STUDENT_SAVE_PATH)
            print(f"  ✅ 保存最优权重 -> {PRETRAIN_STUDENT_SAVE_PATH}")

    print(f"\n预训练蒸馏完成，最优 Loss: {best_loss:.4f}")


if __name__ == "__main__":
    run_pretrain_distill()
