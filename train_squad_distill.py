# train_squad_distill.py
"""
SQuAD 1.1 D组实验：Logits + QA-Focused Attention 蒸馏

用法：
  python train_squad_distill.py           # 训练 + 评估
  python train_squad_distill.py --eval_only  # 只评估已有 checkpoint
"""

import os
import re
import time
import string
import argparse
import collections
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset,load_from_disk
from tqdm import tqdm

from config_squad import (
    TEACHER_NAME, TEACHER_PRETRAIN_NAME, STUDENT_BASE,
    TEACHER_SAVE_PATH, STUDENT_SAVE_PATH,
    PT_STUDENT_SAVE_PATH,DATASET_NAME,
    DATASET_NAME, MAX_LENGTH, DOC_STRIDE, MAX_ANSWER_LENGTH,
    STUDENT_NUM_LAYERS, TEACHER_LAYERS_FOR_DISTILL,
    TEACHER_EPOCHS, TEACHER_BATCH_SIZE, TEACHER_LR,
    TEACHER_WARMUP_RATIO, TEACHER_WEIGHT_DECAY,
    EPOCHS, BATCH_SIZE, LR, WARMUP_RATIO, WEIGHT_DECAY,
    ALPHA, BETA, GAMMA, TEMPERATURE,
)
from models.models_squad import TeacherModel, StudentModel
from utils.pretrain_distill_squad import run_pretrain_distill  # 任务感知预训练蒸馏


# ============================================================
# 数据预处理
# ============================================================

def preprocess_train(examples, tokenizer):
    """SQuAD 训练集预处理：滑窗分割，标注答案 token 位置"""
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map    = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"]   = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx  = sample_map[i]
        answers     = examples["answers"][sample_idx]
        sequence_ids = tokenized.sequence_ids(i)

        ctx_start = next((j for j, s in enumerate(sequence_ids) if s == 1), None)
        ctx_end   = next((j for j, s in reversed(list(enumerate(sequence_ids))) if s == 1), None)

        # SQuAD 1.1 所有问题都有答案
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


def preprocess_validation(examples, tokenizer):
    """SQuAD 验证集预处理：保留 offset_mapping 和 example_id 用于答案还原"""
    questions = [q.strip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        tokenized["example_id"].append(examples["id"][sample_map[i]])

    return tokenized


def get_dataloaders(tokenizer):
    print("加载 SQuAD 1.1 数据集...")
    dataset = load_from_disk(DATASET_NAME)

    train_ds = dataset["train"].map(
        lambda x: preprocess_train(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="处理训练集",
    )
    val_ds = dataset["validation"].map(
        lambda x: preprocess_validation(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="处理验证集",
    )

    # 验证集在 set_format 前保存元数据
    val_meta = {
        "example_id":    val_ds["example_id"],
        "offset_mapping": val_ds["offset_mapping"],
        "token_type_ids": val_ds["token_type_ids"],
    }

    train_ds.set_format("torch")
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    val_raw = dataset["validation"]
    print(f"训练集：{len(train_ds)} 条  验证集：{len(val_ds)} 条")
    return train_loader, val_loader, val_raw, val_meta


# ============================================================
# 蒸馏损失
# ============================================================

def hard_label_loss(student_start, student_end, start_pos, end_pos):
    ce = nn.CrossEntropyLoss()
    return ce(student_start, start_pos) + ce(student_end, end_pos)


def logits_distillation_loss(s_start, s_end, t_start, t_end, T=TEMPERATURE):
    loss_start = F.kl_div(
        F.log_softmax(s_start / T, dim=-1),
        F.softmax(t_start / T, dim=-1),
        reduction="batchmean",
    )
    loss_end = F.kl_div(
        F.log_softmax(s_end / T, dim=-1),
        F.softmax(t_end / T, dim=-1),
        reduction="batchmean",
    )
    return (loss_start + loss_end) * (T ** 2)


def qa_focused_attention_loss(student_attns, teacher_attns,
                              input_ids, token_type_ids,
                              start_pos, end_pos):
    """
    QA Focused Attention Distillation
    SQuAD：token_type_ids==0 为 question（同样排除 [CLS]=101 [SEP]=102）
    """
    mse = nn.MSELoss()
    device = token_type_ids.device
    seq_len = token_type_ids.size(1)

    # question mask（排除特殊 token）
    special = (input_ids == 101) | (input_ids == 102)
    q_mask = ((token_type_ids == 0) & ~special).float()  # (B, L)

    # answer mask（向量化）
    indices = torch.arange(seq_len, device=device).unsqueeze(0)
    s = start_pos.unsqueeze(1)
    e = end_pos.unsqueeze(1)
    a_mask = ((indices >= s) & (indices <= e) & (s <= e)).float()  # (B, L)

    # 重要区域权重 = 2，其余 = 1
    important = torch.clamp(q_mask + a_mask, 0, 1)
    base_w = torch.ones(token_type_ids.size(0), seq_len, seq_len, device=device)
    focus_w = important.unsqueeze(2) * important.unsqueeze(1)
    weight = (base_w + focus_w).unsqueeze(1)  # (B, 1, L, L)

    total_loss = torch.tensor(0.0, device=device)
    for s_idx, t_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_a = student_attns[s_idx]  # (B, H_s, L, L)  H_s=12（bert-base）
        t_a = teacher_attns[t_idx].detach()  # (B, H_t, L, L)  H_t=16（bert-large）

        # 教师 head 数（16）与学生 head 数（12）不同，需要对齐
        # 对教师 attention 在 head 维度做平均池化，降到学生 head 数
        H_s = s_a.size(1)
        H_t = t_a.size(1)
        if H_t != H_s:
            # adaptive_avg_pool1d 对最后一维操作，需要先转置让 H_t 在最后
            # (B, H_t, L, L) -> (B, L*L, H_t) -> pool -> (B, L*L, H_s) -> (B, H_s, L, L)
            B, _, L, _ = t_a.shape
            t_a = t_a.view(B, H_t, L * L)  # (B, H_t, L*L)
            t_a = t_a.permute(0, 2, 1).contiguous()  # (B, L*L, H_t)  ← H_t 移到最后
            t_a = F.adaptive_avg_pool1d(t_a, H_s)  # (B, L*L, H_s)  ← 对 H_t 做池化
            t_a = t_a.permute(0, 2, 1).contiguous()  # (B, H_s, L*L)
            t_a = t_a.view(B, H_s, L, L)  # (B, H_s, L, L)

        s_norm = torch.sqrt(s_a + 1e-6)
        t_norm = torch.sqrt(t_a + 1e-6)
        total_loss = total_loss + mse(s_norm * weight, t_norm * weight)

    return total_loss / len(TEACHER_LAYERS_FOR_DISTILL)


def distillation_loss(student_out, teacher_out, start_pos, end_pos,
                      input_ids, token_type_ids):
    l_hard = hard_label_loss(
        student_out["start_logits"], student_out["end_logits"], start_pos, end_pos
    )
    l_logits = logits_distillation_loss(
        student_out["start_logits"], student_out["end_logits"],
        teacher_out["start_logits"], teacher_out["end_logits"],
    )
    l_attn = qa_focused_attention_loss(
        student_out["attentions"], teacher_out["attentions"],
        input_ids, token_type_ids, start_pos, end_pos,
    )
    total = ALPHA * l_hard + BETA * l_logits + GAMMA * l_attn
    return total, l_hard, l_logits, l_attn


# ============================================================
# EM / F1 评估（SQuAD 官方逻辑）
# ============================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    return normalize_answer(s).split()


def compute_exact(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred, gold):
    pred_tokens = get_tokens(pred)
    gold_tokens = get_tokens(gold)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)


def postprocess_predictions(val_raw, val_meta, all_start_logits, all_end_logits):
    features_per_example = collections.defaultdict(list)
    for i, eid in enumerate(val_meta["example_id"]):
        features_per_example[eid].append(i)

    predictions = {}
    for example in val_raw:
        eid     = example["id"]
        context = example["context"]
        feat_indices = features_per_example[eid]
        best = {"text": "", "score": float("-inf")}

        for fi in feat_indices:
            s_logits = all_start_logits[fi]
            e_logits = all_end_logits[fi]
            offsets  = val_meta["offset_mapping"][fi]
            tt_ids   = val_meta["token_type_ids"][fi]

            # token_type_ids 可能是 tensor，转为 list
            if hasattr(tt_ids, "tolist"):
                tt_ids = tt_ids.tolist()

            ctx_start = next((i for i, t in enumerate(tt_ids) if t == 1), None)
            ctx_end   = next((i for i, t in reversed(list(enumerate(tt_ids))) if t == 1), None)
            if ctx_start is None or ctx_end is None:
                continue

            for si in range(ctx_start, ctx_end + 1):
                for ei in range(si, min(si + MAX_ANSWER_LENGTH, ctx_end + 1)):
                    score = s_logits[si] + e_logits[ei]
                    if score > best["score"]:
                        char_s = offsets[si][0]
                        char_e = offsets[ei][1]
                        best = {"text": context[char_s:char_e], "score": score}

        predictions[eid] = best["text"]
    return predictions


# ============================================================
# 评估函数（含延迟、显存测量）
# ============================================================

def evaluate(model, val_loader, val_raw, val_meta, device, desc="评估"):
    model.eval()
    all_start, all_end = [], []
    times = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / input_ids.size(0))

            # StudentModel 返回 dict，TeacherModel 也返回 dict
            if isinstance(out, dict):
                all_start.append(out["start_logits"].cpu().numpy())
                all_end.append(out["end_logits"].cpu().numpy())
            else:
                all_start.append(out.start_logits.cpu().numpy())
                all_end.append(out.end_logits.cpu().numpy())

    import numpy as np
    all_start = np.concatenate(all_start, axis=0)
    all_end   = np.concatenate(all_end,   axis=0)

    predictions = postprocess_predictions(val_raw, val_meta, all_start, all_end)

    em_scores, f1_scores = [], []
    for example in val_raw:
        eid  = example["id"]
        pred = predictions.get(eid, "")
        golds = example["answers"]["text"]
        em_scores.append(max(compute_exact(pred, g) for g in golds))
        f1_scores.append(max(compute_f1(pred, g) for g in golds))

    peak_gpu = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    return {
        "EM":              round(np.mean(em_scores) * 100, 2),
        "F1":              round(np.mean(f1_scores) * 100, 2),
        "avg_latency_ms":  round(np.mean(times) * 1000, 3),
        "p99_latency_ms":  round(np.percentile(times, 99) * 1000, 3),
        "peak_gpu_mb":     round(peak_gpu, 1),
    }


def get_model_info(model_or_path):
    """统计参数量和模型文件大小"""
    if isinstance(model_or_path, str):
        total = sum(
            os.path.getsize(os.path.join(model_or_path, f))
            for f in os.listdir(model_or_path)
            if f.endswith(".bin") or f.endswith(".safetensors")
        )
        size_mb = total / (1024**2)
        return {"model_size_mb": round(size_mb, 1)}

    params = sum(p.numel() for p in model_or_path.parameters())
    return {"param_count_M": round(params / 1e6, 2)}



# ============================================================
# 教师微调
# ============================================================

def get_teacher_dataloaders(tokenizer):
    """加载 SQuAD，返回教师微调用的训练/验证 DataLoader"""
    print("加载 SQuAD 数据集（教师微调）...")
    dataset = load_from_disk(DATASET_NAME)

    train_ds = dataset["train"].map(
        lambda x: preprocess_train(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="处理训练集（教师）",
    )
    val_ds = dataset["validation"].map(
        lambda x: preprocess_validation(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="处理验证集（教师）",
    )

    val_meta = {
        "example_id":     val_ds["example_id"],
        "offset_mapping": val_ds["offset_mapping"],
        "token_type_ids": val_ds["token_type_ids"],
    }

    train_ds.set_format("torch")
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids"])

    train_loader = DataLoader(train_ds, batch_size=TEACHER_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=TEACHER_BATCH_SIZE, shuffle=False)
    val_raw = dataset["validation"]

    print(f"训练集：{len(train_ds)} 条  验证集：{len(val_ds)} 条")
    return train_loader, val_loader, val_raw, val_meta


def train_teacher():
    """
    在 SQuAD 上微调 bert-large-uncased-whole-word-masking
    微调完成后保存至 TEACHER_SAVE_PATH，供蒸馏阶段使用
    """
    print("=" * 60)
    print("Step 1：教师模型微调")
    print(f"  模型：{TEACHER_PRETRAIN_NAME}")
    print(f"  数据：SQuAD 1.1")
    print(f"  保存：{TEACHER_SAVE_PATH}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PRETRAIN_NAME)
    train_loader, val_loader, val_raw, val_meta = get_teacher_dataloaders(tokenizer)

    # 加载预训练权重，微调模式（不冻结）
    teacher_wrapper = TeacherModel(load_finetuned=False)
    model = teacher_wrapper.model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TEACHER_LR, weight_decay=TEACHER_WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * TEACHER_EPOCHS
    warmup_steps = int(total_steps * TEACHER_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0

    for epoch in range(TEACHER_EPOCHS):
        # ------ 训练 ------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[教师 Epoch {epoch+1}/{TEACHER_EPOCHS}] 训练")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ------ 验证 ------
        # 用 TeacherModel wrapper 做验证（输出 dict，与 evaluate 兼容）
        teacher_wrapper_eval = TeacherModel(load_finetuned=False)
        teacher_wrapper_eval.model = model  # 复用已训练的模型参数
        metrics = evaluate(
            teacher_wrapper_eval, val_loader, val_raw, val_meta,
            device, desc=f"[教师 Epoch {epoch+1}] 验证"
        )

        print(
            f"教师 Epoch {epoch+1} | "
            f"Train Loss:{avg_train_loss:.4f} | "
            f"EM:{metrics['EM']:.2f}% F1:{metrics['F1']:.2f}%"
        )

        # 保存最优 checkpoint
        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            os.makedirs(TEACHER_SAVE_PATH, exist_ok=True)
            model.save_pretrained(TEACHER_SAVE_PATH)
            tokenizer.save_pretrained(TEACHER_SAVE_PATH)
            print(f"  ✅ 保存教师最优模型，F1: {best_f1:.2f}%")

    print(f"\n教师微调完成，最优 F1: {best_f1:.2f}%")
    print(f"Checkpoint 已保存至：{TEACHER_SAVE_PATH}")
    return best_f1


# ============================================================
# 训练主流程
# ============================================================

def train(student_init_mode: str = "bert_base"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    train_loader, val_loader, val_raw, val_meta = get_dataloaders(tokenizer)

    # 教师（加载微调好的 checkpoint）
    teacher = TeacherModel(load_finetuned=True).to(device)
    teacher.eval()

    # 学生（init_mode 由调用方传入）
    student_wrapper = StudentModel(init_mode=student_init_mode)
    student = student_wrapper.model.to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        student.train()
        total_sum = hard_sum = logit_sum = attn_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{EPOCHS}] 训练")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)

            # 教师前向
            teacher_out = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # 学生前向
            student_out = student_wrapper(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss, l_hard, l_logits, l_attn = distillation_loss(
                student_out, teacher_out,
                start_pos, end_pos,
                input_ids, token_type_ids,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_sum += loss.item()
            hard_sum  += l_hard.item()
            logit_sum += l_logits.item()
            attn_sum  += l_attn.item()

            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                hard=f"{l_hard.item():.4f}",
                kd=f"{l_logits.item():.4f}",
                attn=f"{l_attn.item():.4f}",
            )

        n = len(train_loader)
        print(
            f"Epoch {epoch+1} Train | "
            f"Total:{total_sum/n:.4f} Hard:{hard_sum/n:.4f} "
            f"KD:{logit_sum/n:.4f} Attn:{attn_sum/n:.4f}"
        )

        # 验证
        metrics = evaluate(student_wrapper, val_loader, val_raw, val_meta,
                           device, desc=f"[Epoch {epoch+1}] 验证")
        print(
            f"Epoch {epoch+1} Val | "
            f"EM:{metrics['EM']:.2f}% F1:{metrics['F1']:.2f}% "
            f"Latency:{metrics['avg_latency_ms']:.3f}ms GPU:{metrics['peak_gpu_mb']:.0f}MB"
        )

        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            os.makedirs(STUDENT_SAVE_PATH, exist_ok=True)
            student.save_pretrained(STUDENT_SAVE_PATH)
            tokenizer.save_pretrained(STUDENT_SAVE_PATH)
            print(f"  ✅ 保存最优模型，F1: {best_f1:.2f}%")

    print(f"\n训练完成，最优 F1: {best_f1:.2f}%")
    return best_f1


# ============================================================
# 最终评估报告（教师 vs 学生对比）
# ============================================================

def final_report():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    _, val_loader, val_raw, val_meta = get_dataloaders(tokenizer)

    results = []

    # 教师评估
    print("\n评估教师模型...")
    teacher = TeacherModel(load_finetuned=True).to(device)
    t_metrics = evaluate(teacher, val_loader, val_raw, val_meta, device, desc="教师推理")
    t_info    = get_model_info(teacher)
    results.append({
        "模型":         "教师 (bert-large-wwm-squad)",
        "参数量(M)":    round(sum(p.numel() for p in teacher.parameters()) / 1e6, 1),
        **t_metrics,
    })
    del teacher
    torch.cuda.empty_cache()

    # 学生评估
    if os.path.exists(STUDENT_SAVE_PATH):
        print("\n评估学生模型（蒸馏后）...")
        from transformers import AutoModelForQuestionAnswering
        config = __import__("transformers").AutoConfig.from_pretrained(STUDENT_SAVE_PATH)
        config.output_attentions = True
        config.output_hidden_states = True
        stu_model = AutoModelForQuestionAnswering.from_pretrained(STUDENT_SAVE_PATH, config=config)

        class _Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 output_attentions=True, output_hidden_states=True)
                return {"start_logits": out.start_logits, "end_logits": out.end_logits,
                        "attentions": out.attentions, "hidden_states": out.hidden_states}

        wrapped = _Wrapper(stu_model).to(device)
        s_metrics = evaluate(wrapped, val_loader, val_raw, val_meta, device, desc="学生推理")
        results.append({
            "模型":         "学生 (bert-base 6层蒸馏)",
            "参数量(M)":    round(sum(p.numel() for p in stu_model.parameters()) / 1e6, 1),
            **s_metrics,
        })
    else:
        print(f"未找到学生 checkpoint：{STUDENT_SAVE_PATH}，跳过学生评估")

    # 打印对比表
    print("\n" + "=" * 80)
    print("SQuAD 实验对比报告（D组：Logits + QA-Focused Attention 蒸馏）")
    print("=" * 80)
    header = f"{'模型':<30} {'EM%':>7} {'F1%':>7} {'延迟(ms)':>10} {'P99(ms)':>9} {'参数(M)':>8} {'峰值显存(MB)':>12}"
    print(header)
    print("-" * 80)
    for r in results:
        print(
            f"{r['模型']:<30} "
            f"{r['EM']:>7.2f} {r['F1']:>7.2f} "
            f"{r['avg_latency_ms']:>10.3f} {r['p99_latency_ms']:>9.3f} "
            f"{r['参数量(M)']:>8.1f} {r['peak_gpu_mb']:>12.1f}"
        )

    if len(results) == 2:
        t, s = results[0], results[1]
        print("\n相对教师的压缩与保留率：")
        print(f"  F1 保留率：{s['F1']/t['F1']*100:.1f}%")
        print(f"  参数压缩：{t['参数量(M)']/s['参数量(M)']:.1f}x")
        print(f"  推理加速：{t['avg_latency_ms']/s['avg_latency_ms']:.1f}x")
        print(f"  显存减少：{(1 - s['peak_gpu_mb']/t['peak_gpu_mb'])*100:.1f}%")

    print("=" * 80)

    # 保存结果
    os.makedirs("./results_squad", exist_ok=True)
    with open("./results_squad/report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("结果已保存至 ./results_squad/report.json")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only",           action="store_true",
                        help="只做最终评估报告，不训练")
    parser.add_argument("--skip_teacher_train",  action="store_true",
                        help="跳过教师微调（已有 checkpoint 时使用）")
    parser.add_argument("--skip_pretrain_distill", action="store_true",
                        help="跳过预训练蒸馏（已有权重时使用）")
    parser.add_argument("--teacher_only",        action="store_true",
                        help="只微调教师")
    parser.add_argument("--pretrain_only",       action="store_true",
                        help="只做预训练蒸馏")
    parser.add_argument("--two_stage",           action="store_true",
                        help="两阶段蒸馏：预训练蒸馏 + 微调蒸馏（推荐）")
    parser.add_argument("--exit", action="store_true",help="查看文件是否存在")
    args = parser.parse_args()

    if args.eval_only:
        final_report()

    elif args.teacher_only:
        train_teacher()

    elif args.pretrain_only:
        run_pretrain_distill()

    elif args.two_stage:
        # ====== 两阶段蒸馏完整流程 ======
        # Step 1：教师微调（bert-large-uncased 在 SQuAD 上微调）
        if os.path.exists(TEACHER_SAVE_PATH):
            print(f"跳过教师微调，使用已有 checkpoint：{TEACHER_SAVE_PATH}")
        else:
            train_teacher()

        # Step 2：任务感知预训练蒸馏（英文 Wikipedia + SQuAD QA 混合）
        if os.path.exists(PT_STUDENT_SAVE_PATH):
            print(f"跳过预训练蒸馏，使用已有权重：{PT_STUDENT_SAVE_PATH}")
        else:
            run_pretrain_distill()

        # Step 3：微调阶段蒸馏（学生从预训练蒸馏权重初始化，Logits + QA-Attention）
        print("\n>>> 微调阶段蒸馏（学生从预训练蒸馏权重初始化）")
        train(student_init_mode="pretrain_distill")

        # Step 4：评估报告
        final_report()

    elif args.exit:
        if os.path.exists(PT_STUDENT_SAVE_PATH):
            print("存在")
        else:
            print("不存在")
    else:
        # ====== 单阶段蒸馏（默认）======
        # Step 1：教师微调
        if args.skip_teacher_train and os.path.exists(TEACHER_SAVE_PATH):
            print(f"跳过教师微调，使用已有 checkpoint：{TEACHER_SAVE_PATH}")
        else:
            train_teacher()

        # Step 2：微调阶段蒸馏（从 bert-base 初始化）
        train(student_init_mode="bert_base")

        # Step 3：评估报告
        final_report()

    os.system("/usr/bin/shutdown")