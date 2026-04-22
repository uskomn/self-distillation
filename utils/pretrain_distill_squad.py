# pretrain_distill_squad.py
"""
预训练阶段：任务感知预训练蒸馏（Task-Aware Pretraining Distillation）
英文版，完全对应中文版 pretrain_distill.py 的设计思路

损失组成：
  Wikipedia batch : L_MLM + L_logits + L_hidden + L_attn  （通用语言表征）
                     L_logits：学生与教师 MLM logits 的 KL 散度（软标签蒸馏）
  QA batch        : L_QA_hard + L_QA_kd           （任务意识注入）

两种 batch 交替出现，比例由 PT_QA_INTERLEAVE_EVERY 控制：
  每 PT_QA_INTERLEAVE_EVERY 个 Wiki batch 插入 1 个 QA batch

教师模型：
  - Wikipedia 阶段：bert-base-uncased 预训练权重（提供隐层/attention 蒸馏信号）
                   对应中文版的 hfl/chinese-roberta-wwm-ext
  - QA 阶段：      bert-large-uncased（SQuAD 微调后，提供 QA logits）
                   对应中文版的 CMRC2018 微调后教师

学生模型：
  - 6层 bert-base-uncased
  - 从 PT_TEACHER_NAME（bert-base-uncased）偶数层 [0,2,4,6,8,10] 初始化
  - 附加临时 QA head 接收任务信号，微调阶段丢弃

完成后保存至 PT_STUDENT_SAVE_PATH，供微调阶段蒸馏初始化学生使用。
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
from datasets import load_dataset, Dataset,load_from_disk
from tqdm import tqdm

try:
    import mwxml
    HAS_MWXML = True
except ImportError:
    HAS_MWXML = False

from config_squad import (
    PT_TEACHER_NAME,
    TEACHER_SAVE_PATH,          # SQuAD 微调后教师，用于 QA logits
    STUDENT_BASE,
    STUDENT_NUM_LAYERS,
    PT_TEACHER_LAYERS,
    PT_STUDENT_SAVE_PATH,
    PT_DATA_SOURCE,
    PT_WIKI_NAME, PT_WIKI_CONFIG,
    PT_LOCAL_XML_PATH, PT_LOCAL_CACHE_DIR,
    PT_MAX_SAMPLES, PT_MLM_PROB,
    PT_EPOCHS, PT_BATCH_SIZE, PT_LR,
    PT_WARMUP_RATIO, PT_WEIGHT_DECAY,
    PT_LAMBDA_MLM, PT_LAMBDA_LOGITS, PT_LAMBDA_HIDDEN, PT_LAMBDA_ATTN,
    PT_QA_INTERLEAVE_EVERY,
    PT_LAMBDA_QA_HARD, PT_LAMBDA_QA_KD,
    PT_USE_QA_KD,PT_LAMBDA_EMBEDDING,
    MAX_LENGTH, DOC_STRIDE,
    TEMPERATURE,
    DATASET_NAME,
)


# ============================================================
# 教师模型
# ============================================================

class PretrainTeacher(nn.Module):
    """
    预训练阶段教师：bert-base-uncased 原始预训练权重，冻结
    使用 BertForMaskedLM（而非 AutoModel），以便同时输出 MLM logits 供 logits 蒸馏
    输出：hidden_states、attentions、mlm_logits
    hidden_size=768，与学生相同，无需投影层
    """
    def __init__(self):
        super().__init__()
        print(f"加载预训练教师（BertForMaskedLM）：{PT_TEACHER_NAME}（冻结）")
        self.model = BertForMaskedLM.from_pretrained(PT_TEACHER_NAME)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                output_attentions=True,
            )
        return outputs.hidden_states, outputs.attentions, outputs.logits
        # hidden_states : tuple 13 x (B, L, 768)
        # attentions    : tuple 12 x (B, 12, L, L)
        # logits        : (B, L, vocab_size)  MLM 预测 logits，用于软标签 KL 蒸馏


class QATeacher(nn.Module):
    """
    QA 阶段教师：bert-large-uncased（SQuAD 微调后），冻结
    提供 QA logits 蒸馏信号
    对应中文版的 CMRC2018 微调后教师
    """
    def __init__(self):
        super().__init__()
        print(f"加载 QA 教师：{TEACHER_SAVE_PATH}")
        config = AutoConfig.from_pretrained(TEACHER_SAVE_PATH)
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
    """
    预训练阶段学生：6层 BertForMaskedLM
    从 PT_TEACHER_NAME（bert-base-uncased）偶数层 PT_TEACHER_LAYERS 初始化
    对应中文版从 hfl/chinese-roberta-wwm-ext 偶数层初始化
    """
    def __init__(self):
        super().__init__()
        config = BertConfig.from_pretrained(STUDENT_BASE)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        self.model = BertForMaskedLM(config)
        self._init_from_teacher()

    def _init_from_teacher(self):
        print(f"从教师偶数层 {PT_TEACHER_LAYERS} 初始化学生...")
        teacher = AutoModel.from_pretrained(PT_TEACHER_NAME)
        # embedding 层
        self.model.bert.embeddings.load_state_dict(teacher.embeddings.state_dict())
        # encoder 偶数层：[0,2,4,6,8,10] -> 学生 [0,1,2,3,4,5]
        for student_idx, teacher_idx in enumerate(PT_TEACHER_LAYERS):
            self.model.bert.encoder.layer[student_idx].load_state_dict(
                teacher.encoder.layer[teacher_idx].state_dict()
            )
        del teacher
        torch.cuda.empty_cache()
        print("学生初始化完成。")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,  # 显式传入，避免版本差异
            output_attentions=True,
        )

    def get_bert_output(self, input_ids, attention_mask, token_type_ids=None):
        """
        QA batch 时取 bert encoder 输出，不走 MLM head
        对应中文版的 get_bert_output
        """
        return self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )


# ============================================================
# QA head（任务感知阶段临时使用）
# ============================================================

class QAHead(nn.Module):
    """
    轻量 QA head，接在学生 bert encoder 后
    预训练阶段接收 QA 任务监督信号，微调阶段丢弃
    对应中文版 QAHead
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

def wiki_distill_loss(student_outputs, teacher_hidden, teacher_attn, teacher_logits,
                      attention_mask,
                      lambda_mlm=PT_LAMBDA_MLM,
                      lambda_logits=PT_LAMBDA_LOGITS,
                      lambda_embedding=PT_LAMBDA_EMBEDDING,
                      lambda_hidden=PT_LAMBDA_HIDDEN,
                      lambda_attn=PT_LAMBDA_ATTN,
                      temperature=TEMPERATURE):
    """
    Wikipedia batch 五项损失：MLM + Logits KL + Embedding + 隐层 + Attention

    L_MLM       : 学生自身 Masked Language Modeling 交叉熵
    L_logits    : 学生与教师 MLM logits 的 KL 散度（软标签蒸馏）
                  只在非 padding token 上计算
    L_embedding : Embedding 层输出对齐
                  教师 embedding(0) 与学生 embedding(0) 做 cosine MSE
                  让学生从预训练开始就建立与教师一致的词表示空间
                  同时用 attention_mask 只对真实 token 位置计算，排除 padding 干扰
    L_hidden    : 逐层隐层 cosine 归一化 MSE（encoder 层，索引从 1 开始）
    L_attn      : 逐层 attention sqrt 归一化 MSE
    """
    mse = nn.MSELoss()

    l_mlm = student_outputs.loss
    if l_mlm is None:
        raise ValueError("MLM loss 为 None，请确认传入了 labels")
    if student_outputs.hidden_states is None or student_outputs.attentions is None:
        raise ValueError("hidden_states 或 attentions 为 None，请确认 forward 传入了对应参数")

    mask = attention_mask.float()  # (B, L)

    # ------ L_logits：MLM logits KL 散度 ------
    # 只在非 padding token 上计算，避免 padding 噪声
    T = temperature
    s_log_prob = F.log_softmax(student_outputs.logits / T, dim=-1)  # (B, L, V)
    t_prob = F.softmax(teacher_logits.detach() / T, dim=-1)  # (B, L, V)
    kl_per_token = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)  # (B, L)
    l_logits = (kl_per_token * mask).sum() / mask.sum() * (T ** 2)

    # ------ L_embedding：Embedding 层输出对齐 ------
    # hidden_states[0] 是 embedding 层输出（position + token + segment embedding 之和）
    # 用 cosine 归一化后做 MSE，与隐层对齐方式保持一致
    # 额外用 attention_mask 排除 padding 位置，只对真实 token 计算
    s_emb = student_outputs.hidden_states[0]  # (B, L, 768)
    t_emb = teacher_hidden[0].detach()  # (B, L, 768)
    s_emb_norm = F.normalize(s_emb, dim=-1)  # (B, L, 768)
    t_emb_norm = F.normalize(t_emb, dim=-1)  # (B, L, 768)
    # 只在非 padding 位置计算 MSE
    emb_diff = (s_emb_norm - t_emb_norm) ** 2  # (B, L, 768)
    l_embedding = (emb_diff.mean(dim=-1) * mask).sum() / mask.sum()

    # ------ L_hidden：逐层隐层 cosine 归一化 MSE ------
    # hidden_states[1..6] 对应 encoder 层 0..5
    l_hidden = torch.tensor(0.0, device=l_mlm.device)
    for s_idx, t_idx in enumerate(PT_TEACHER_LAYERS):
        s_h = student_outputs.hidden_states[s_idx + 1]  # (B, L, 768)
        t_h = teacher_hidden[t_idx + 1].detach()  # (B, L, 768)
        l_hidden = l_hidden + mse(F.normalize(s_h, dim=-1), F.normalize(t_h, dim=-1))
    l_hidden = l_hidden / len(PT_TEACHER_LAYERS)

    # ------ L_attn：逐层 attention sqrt 归一化 MSE ------
    l_attn = torch.tensor(0.0, device=l_mlm.device)
    for s_idx, t_idx in enumerate(PT_TEACHER_LAYERS):
        s_a = student_outputs.attentions[s_idx]  # (B, 12, L, L)
        t_a = teacher_attn[t_idx].detach()  # (B, 12, L, L)
        l_attn = l_attn + mse(torch.sqrt(s_a + 1e-6), torch.sqrt(t_a + 1e-6))
    l_attn = l_attn / len(PT_TEACHER_LAYERS)

    total = (lambda_mlm * l_mlm
             + lambda_logits * l_logits
             + lambda_embedding * l_embedding
             + lambda_hidden * l_hidden
             + lambda_attn * l_attn)
    return total, l_mlm, l_logits, l_embedding, l_hidden, l_attn


def qa_task_loss(student_start, student_end,
                 start_positions, end_positions,
                 teacher_start=None, teacher_end=None,
                 lambda_hard=PT_LAMBDA_QA_HARD,
                 lambda_kd=PT_LAMBDA_QA_KD,
                 temperature=TEMPERATURE):
    """
    QA batch 损失：hard label CE + 教师 logits KD
    teacher_start/end 为 None 时只用 hard label
    与中文版 qa_task_loss 完全对称
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
# 数据加载：英文 Wikipedia
# ============================================================

def _clean_wiki_en(raw):
    if raw.strip().lower().startswith("#redirect"):
        return ""
    text = re.sub(r"\{\{[^}]*\}\}", "", raw)
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"=+([^=]+)=+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_local_xml():
    cache_file = os.path.join(PT_LOCAL_CACHE_DIR, "texts_en.jsonl")
    if os.path.exists(cache_file):
        print(f"发现缓存：{cache_file}")
        texts = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="读取缓存"):
                texts.append(json.loads(line)["text"])
                if PT_MAX_SAMPLES and len(texts) >= PT_MAX_SAMPLES:
                    break
        print(f"缓存加载完成：{len(texts)} 条")
        return texts

    os.makedirs(PT_LOCAL_CACHE_DIR, exist_ok=True)
    print(f"解析本地 XML：{PT_LOCAL_XML_PATH}")
    texts = []

    if HAS_MWXML:
        dump = mwxml.Dump.from_file(bz2.open(PT_LOCAL_XML_PATH, "rb"))
        with open(cache_file, "w", encoding="utf-8") as cf:
            for page in tqdm(dump.pages, desc="mwxml 解析"):
                if page.namespace != 0:
                    continue
                for rev in page:
                    text = _clean_wiki_en(rev.text or "")
                    if len(text) >= 100:
                        texts.append(text)
                        cf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    break
                if PT_MAX_SAMPLES and len(texts) >= PT_MAX_SAMPLES:
                    break
    else:
        print("mwxml 未安装，使用正则解析")
        with open(cache_file, "w", encoding="utf-8") as cf:
            with bz2.open(PT_LOCAL_XML_PATH, "rt", encoding="utf-8") as f:
                in_text, buf = False, []
                for line in tqdm(f, desc="正则解析"):
                    if "<text" in line:
                        m = re.search(r"<text[^>]*>(.*?)</text>", line, re.DOTALL)
                        if m:
                            text = _clean_wiki_en(m.group(1))
                            if len(text) >= 100:
                                texts.append(text)
                                cf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        else:
                            m2 = re.search(r"<text[^>]*>(.*)", line, re.DOTALL)
                            buf = [m2.group(1)] if m2 else []
                            in_text = True
                    elif in_text:
                        if "</text>" in line:
                            buf.append(line[:line.index("</text>")])
                            text = _clean_wiki_en("".join(buf))
                            if len(text) >= 100:
                                texts.append(text)
                                cf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                            in_text, buf = False, []
                        else:
                            buf.append(line)
                    if PT_MAX_SAMPLES and len(texts) >= PT_MAX_SAMPLES:
                        break

    print(f"解析完成：{len(texts)} 条")
    return texts


def _load_raw_wiki():
    source = PT_DATA_SOURCE.lower()
    split  = "train[:{}]".format(PT_MAX_SAMPLES)
    if source == "wikimedia":
        print(f"加载在线 Wikipedia：{PT_WIKI_NAME} / {PT_WIKI_CONFIG}")
        ds = load_dataset(PT_WIKI_NAME, PT_WIKI_CONFIG, split=split, trust_remote_code=True)
        return ds.select_columns(["text"])
    elif source == "local_xml":
        texts = _parse_local_xml()
        return Dataset.from_dict({"text": texts[:PT_MAX_SAMPLES] if PT_MAX_SAMPLES else texts})
    else:
        raise ValueError(f"未知数据源：{source}，请设为 wikimedia 或 local_xml")


def get_wiki_dataloader(tokenizer):
    print(f"加载 Wikipedia 数据，来源：{PT_DATA_SOURCE}，上限：{PT_MAX_SAMPLES}")
    dataset = _load_raw_wiki()
    print(f"原始数据：{len(dataset)} 条")

    def tokenize_fn(examples):
        texts = [t if (t and t.strip()) else "empty" for t in examples["text"]]
        return tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length")

    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing Wikipedia (EN)",
    )
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=PT_MLM_PROB
    )
    loader = DataLoader(tokenized, batch_size=PT_BATCH_SIZE, shuffle=True, collate_fn=collator)
    print(f"Wikipedia 数据就绪：{len(tokenized)} 条，{len(loader)} 个 batch")
    return loader


# ============================================================
# 数据加载：SQuAD QA（对应中文版 CMRC2018）
# ============================================================

def _preprocess_squad_qa(examples, tokenizer):
    """
    SQuAD 预处理，与中文版 _preprocess_qa 对称
    对应中文版中 CMRC2018 的处理逻辑
    """
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions, examples["context"],
        max_length=MAX_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length",
    )
    sample_map     = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"]   = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx   = sample_map[i]
        answers      = examples["answers"][sample_idx]
        sequence_ids = tokenized.sequence_ids(i)

        ctx_start = next((j for j, s in enumerate(sequence_ids) if s == 1), None)
        ctx_end   = next((j for j, s in reversed(list(enumerate(sequence_ids))) if s == 1), None)

        ans_start_char = answers["answer_start"][0]
        ans_text       = answers["text"][0]
        ans_end_char   = ans_start_char + len(ans_text)

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
    """
    加载 SQuAD 训练集用于任务感知阶段
    对应中文版的 get_qa_dataloader（加载 CMRC2018）
    """
    print("加载 SQuAD QA 数据（任务感知用）...")
    dataset = load_from_disk(DATASET_NAME)["train"]
    tokenized = dataset.map(
        lambda x: _preprocess_squad_qa(x, tokenizer),
        batched=True, remove_columns=dataset.column_names,
        desc="Tokenizing SQuAD (QA aware)",
    )
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=PT_BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"SQuAD QA 数据就绪：{len(tokenized)} 条，{len(loader)} 个 batch")
    return loader


# ============================================================
# 主训练流程
# ============================================================

def run_pretrain_distill():
    print("=" * 65)
    print("预训练阶段：任务感知预训练蒸馏（英文版）")
    print(f"  Wiki 教师 ：{PT_TEACHER_NAME}（bert-base，12层，冻结）")
    print(f"  QA 教师   ：{TEACHER_SAVE_PATH}（bert-large SQuAD 微调后）")
    print(f"  学生      ：{STUDENT_BASE} 6层，偶数层 {PT_TEACHER_LAYERS} 初始化")
    print(f"  Wiki batch：每 {PT_QA_INTERLEAVE_EVERY} 个插入 1 个 QA batch")
    print(f"  QA 损失   ：hard={PT_LAMBDA_QA_HARD}  kd={'on' if PT_USE_QA_KD else 'off'}({PT_LAMBDA_QA_KD})")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(PT_TEACHER_NAME)

    # ---- 教师 ----
    pretrain_teacher = PretrainTeacher().to(device)
    pretrain_teacher.eval()

    qa_teacher = None
    if PT_USE_QA_KD and os.path.exists(TEACHER_SAVE_PATH):
        qa_teacher = QATeacher().to(device)
        qa_teacher.eval()
    else:
        print("QA 教师不可用（checkpoint 不存在或 PT_USE_QA_KD=False），QA 阶段只用 hard label")

    # ---- 学生 ----
    student_wrapper = PretrainStudent()
    student = student_wrapper.model.to(device)

    # QA head（临时，预训练后丢弃）
    qa_head = QAHead(student.config.hidden_size).to(device)

    # ---- 数据 ----
    wiki_loader = get_wiki_dataloader(tokenizer)
    qa_loader   = get_qa_dataloader(tokenizer)
    qa_iter     = cycle(qa_loader)   # QA 数据量少，循环使用

    # ---- 优化器（学生 bert + qa_head）----
    all_params = list(student.parameters()) + list(qa_head.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=PT_LR, weight_decay=PT_WEIGHT_DECAY)
    total_steps  = len(wiki_loader) * PT_EPOCHS
    warmup_steps = int(total_steps * PT_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_loss = float("inf")

    for epoch in range(PT_EPOCHS):
        student.train()
        qa_head.train()

        stats = {k: 0.0 for k in [
            "total", "mlm", "logits", "hidden", "attn",
            "qa_hard", "qa_kd", "wiki_steps", "qa_steps"
        ]}

        pbar = tqdm(wiki_loader, desc=f"[任务感知预训练蒸馏 Epoch {epoch+1}/{PT_EPOCHS}]")

        for wiki_step, wiki_batch in enumerate(pbar):

            # ========== Wikipedia batch ==========
            input_ids      = wiki_batch["input_ids"].to(device)
            attention_mask = wiki_batch["attention_mask"].to(device)
            labels         = wiki_batch["labels"].to(device)
            token_type_ids = wiki_batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            t_hidden, t_attn, t_logits = pretrain_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            s_out = student_wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss_wiki, l_mlm, l_logits, l_hidden, l_attn = wiki_distill_loss(
                s_out, t_hidden, t_attn, t_logits,
                attention_mask=attention_mask,
            )

            loss_wiki.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            stats["total"]      += loss_wiki.item()
            stats["mlm"]        += l_mlm.item()
            stats["logits"]     += l_logits.item()
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

                # 学生 bert encoder 输出（不走 MLM head）
                bert_out = student_wrapper.get_bert_output(
                    input_ids=qa_input_ids,
                    attention_mask=qa_attention_mask,
                    token_type_ids=qa_token_type_ids,
                )
                s_start, s_end = qa_head(bert_out.last_hidden_state)

                # QA 教师 logits
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

            pbar.set_postfix(
                wiki=f"{loss_wiki.item():.3f}",
                mlm=f"{l_mlm.item():.3f}",
                lgts=f"{l_logits.item():.3f}",
                qa_h=f"{stats['qa_hard'] / max(stats['qa_steps'], 1):.3f}",
            )

        # Epoch 汇总
        ws  = stats["wiki_steps"]
        qs  = max(stats["qa_steps"], 1)
        avg = stats["total"] / ws
        print(
            f"Epoch {epoch+1} | "
            f"Wiki Total:{avg:.4f} MLM:{stats['mlm']/ws:.4f} "
            f"Logits:{stats['logits']/ws:.4f} "
            f"Hidden:{stats['hidden']/ws:.4f} Attn:{stats['attn']/ws:.4f} | "
            f"QA Hard:{stats['qa_hard']/qs:.4f} QA KD:{stats['qa_kd']/qs:.4f}"
        )

        # 保存最优权重（保存完整 BertForMaskedLM，qa_head 不保存）
        if avg < best_loss:
            best_loss = avg
            os.makedirs(PT_STUDENT_SAVE_PATH, exist_ok=True)
            student.save_pretrained(PT_STUDENT_SAVE_PATH)
            tokenizer.save_pretrained(PT_STUDENT_SAVE_PATH)
            print(f"   保存最优权重 Loss:{avg:.4f} -> {PT_STUDENT_SAVE_PATH}")

    print(f"\n任务感知预训练蒸馏完成，最优 Loss: {best_loss:.4f}")
    print("注：qa_head 为临时模块，未保存；微调阶段使用 BertForQuestionAnswering 替代 MLM head")


if __name__ == "__main__":
    run_pretrain_distill()