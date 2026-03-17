# experiments.py
"""
实验方案：
  A组：教师模型（Baseline）         hfl/chinese-roberta-wwm-ext 直接微调
  B组：纯 Hard Label 蒸馏           只用 CE loss，无软标签/attention 蒸馏
  C组：Logits 蒸馏                  KD loss + Hard label
  D组：Logits + QA-Focused Attention 蒸馏（完整方案）

用法：
  python experiments.py --exp all        # 跑全部4组
  python experiments.py --exp A          # 只跑 A 组
  python experiments.py --exp B,C        # 跑 B、C 组
  python experiments.py --report_only    # 仅汇总已有结果
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from tqdm import tqdm

from config import (
    TEACHER_NAME, TEACHER_SAVE_PATH,
    MAX_LENGTH, DOC_STRIDE,
    TEACHER_EPOCHS, TEACHER_BATCH_SIZE, TEACHER_LR,
    TEACHER_WARMUP_RATIO, TEACHER_WEIGHT_DECAY,
    STUDENT_EPOCHS, STUDENT_BATCH_SIZE, STUDENT_LR,
    STUDENT_WARMUP_RATIO, STUDENT_WEIGHT_DECAY,
    STUDENT_NUM_LAYERS, TEACHER_LAYERS_FOR_DISTILL,
    PRETRAIN_STUDENT_SAVE_PATH,
    ALPHA, BETA, GAMMA, TEMPERATURE,
)
from models.distill_loss import hard_label_loss, logits_distillation_loss, qa_focused_attention_loss
from utils.evaluate import evaluate_model
from utils.pretrain_distill import run_pretrain_distill


# ============================================================
# 路径配置
# ============================================================

EXPERIMENT_PATHS = {
    "A": "/root/autodl-tmp/checkpoints/exp_A_teacher",
    "B": "/root/autodl-tmp/checkpoints/exp_B_hard_only",
    "C": "/root/autodl-tmp/checkpoints/exp_C_logits_kd",
    "D": "/root/autodl-tmp/checkpoints/exp_D_logits_attn_kd",
    "E": "/root/autodl-tmp/checkpoints/exp_E_two_stage_distill",  # 两阶段蒸馏
}

RESULTS_FILE = "/root/autodl-tmp/results/all_experiments.json"


# ============================================================
# 数据预处理（训练集）
# ============================================================

def preprocess_train(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions, contexts,
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]

        sequence_ids = tokenized.sequence_ids(i)
        # 找 context 范围
        context_start = next((j for j, s in enumerate(sequence_ids) if s == 1), None)
        context_end = next((j for j, s in reversed(list(enumerate(sequence_ids))) if s == 1), None)

        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_char = answer_start_char + len(answer_text)

        if (context_start is None or context_end is None or
                offsets[context_start][0] > answer_end_char or
                offsets[context_end][1] < answer_start_char):
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                token_start += 1
            start_position = token_start - 1

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end_char:
                token_end -= 1
            end_position = token_end + 1

            tokenized["start_positions"].append(start_position)
            tokenized["end_positions"].append(end_position)

    return tokenized


def get_dataloaders(tokenizer, batch_size):
    dataset = load_from_disk("/root/autodl-tmp/cmrc2018")

    train_dataset = dataset["train"].map(
        lambda x: preprocess_train(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["validation"].map(
        lambda x: preprocess_train(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    )


# ============================================================
# 通用工具
# ============================================================

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("使用 CPU")
    return d


def make_optimizer_scheduler(model, train_loader, lr, warmup_ratio, weight_decay, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def save_best(model, tokenizer, save_path, val_loss, best_val_loss, epoch):
    if val_loss < best_val_loss:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Epoch {epoch+1} 保存最优模型，Val Loss: {val_loss:.4f}")
        return val_loss
    return best_val_loss


def build_student_model_from_teacher(device, init_mode="finetune_teacher"):
    """
    构建学生 BertForQuestionAnswering
    init_mode:
      "finetune_teacher"  : 从微调教师偶数层初始化（B/C/D 组）
      "pretrain_distill"  : 从预训练蒸馏权重初始化（E 组）
    """
    from models.distill_models import StudentModel
    wrapper = StudentModel(init_mode=init_mode)
    return wrapper.model.to(device)


def load_finetuned_teacher(device):
    """加载已微调好的教师模型（推理模式）"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(TEACHER_SAVE_PATH)
    config.output_attentions = True
    config.output_hidden_states = True
    teacher = AutoModelForQuestionAnswering.from_pretrained(TEACHER_SAVE_PATH, config=config)
    teacher.eval()
    return teacher.to(device)


# ============================================================
# A组：教师模型微调（Baseline）
# ============================================================

def run_experiment_A():
    print("\n" + "="*60)
    print("A组：教师模型微调（Baseline）")
    print("="*60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    train_loader, val_loader = get_dataloaders(tokenizer, TEACHER_BATCH_SIZE)

    model = AutoModelForQuestionAnswering.from_pretrained(TEACHER_NAME).to(device)
    optimizer, scheduler = make_optimizer_scheduler(
        model, train_loader, TEACHER_LR, TEACHER_WARMUP_RATIO, TEACHER_WEIGHT_DECAY, TEACHER_EPOCHS
    )

    save_path = EXPERIMENT_PATHS["A"]
    # 同时也保存到 TEACHER_SAVE_PATH 供后续蒸馏实验使用
    best_val_loss = float("inf")

    for epoch in range(TEACHER_EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"[A组 Epoch {epoch+1}/{TEACHER_EPOCHS}] 训练"):
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                start_positions=batch["start_positions"].to(device),
                end_positions=batch["end_positions"].to(device),
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    start_positions=batch["start_positions"].to(device),
                    end_positions=batch["end_positions"].to(device),
                )
                val_loss += out.loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"A组 Epoch {epoch+1} Val Loss: {avg_val:.4f}")
        best_val_loss = save_best(model, tokenizer, save_path, avg_val, best_val_loss, epoch)

    # 同步到教师 checkpoint，供 B/C/D 组使用
    if not os.path.exists(TEACHER_SAVE_PATH):
        os.makedirs(TEACHER_SAVE_PATH, exist_ok=True)
        model_final = AutoModelForQuestionAnswering.from_pretrained(save_path)
        model_final.save_pretrained(TEACHER_SAVE_PATH)
        tokenizer.save_pretrained(TEACHER_SAVE_PATH)
        print(f"教师 checkpoint 同步至：{TEACHER_SAVE_PATH}")


# ============================================================
# B组：纯 Hard Label 蒸馏（仅 CE loss，无软标签）
# ============================================================

def run_experiment_B():
    print("\n" + "="*60)
    print("B组：纯 Hard Label 蒸馏（无软标签/Attention）")
    print("="*60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_SAVE_PATH)
    train_loader, val_loader = get_dataloaders(tokenizer, STUDENT_BATCH_SIZE)

    student = build_student_model_from_teacher(device)
    optimizer, scheduler = make_optimizer_scheduler(
        student, train_loader, STUDENT_LR, STUDENT_WARMUP_RATIO, STUDENT_WEIGHT_DECAY, STUDENT_EPOCHS
    )

    save_path = EXPERIMENT_PATHS["B"]
    best_val_loss = float("inf")

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        for batch in tqdm(train_loader, desc=f"[B组 Epoch {epoch+1}/{STUDENT_EPOCHS}] 训练"):
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            out = student(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )

            loss = hard_label_loss(out.start_logits, out.end_logits, start_pos, end_pos)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )
                l = hard_label_loss(
                    out.start_logits, out.end_logits,
                    batch["start_positions"].to(device),
                    batch["end_positions"].to(device),
                )
                val_loss += l.item()

        avg_val = val_loss / len(val_loader)
        print(f"B组 Epoch {epoch+1} Val Loss: {avg_val:.4f}")
        best_val_loss = save_best(student, tokenizer, save_path, avg_val, best_val_loss, epoch)


# ============================================================
# C组：Logits 蒸馏（KD loss + Hard label）
# ============================================================

def run_experiment_C():
    print("\n" + "="*60)
    print("C组：Logits 蒸馏（KD + Hard Label）")
    print("="*60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_SAVE_PATH)
    train_loader, val_loader = get_dataloaders(tokenizer, STUDENT_BATCH_SIZE)

    teacher = load_finetuned_teacher(device)
    student = build_student_model_from_teacher(device)
    optimizer, scheduler = make_optimizer_scheduler(
        student, train_loader, STUDENT_LR, STUDENT_WARMUP_RATIO, STUDENT_WEIGHT_DECAY, STUDENT_EPOCHS
    )

    save_path = EXPERIMENT_PATHS["C"]
    best_val_loss = float("inf")

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        for batch in tqdm(train_loader, desc=f"[C组 Epoch {epoch+1}/{STUDENT_EPOCHS}] 训练"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            s_out = student(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            l_hard = hard_label_loss(s_out.start_logits, s_out.end_logits, start_pos, end_pos)
            l_kd = logits_distillation_loss(
                s_out.start_logits, s_out.end_logits,
                t_out.start_logits, t_out.end_logits,
                TEMPERATURE
            )
            loss = ALPHA * l_hard + BETA * l_kd
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s_out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )
                val_loss += hard_label_loss(
                    s_out.start_logits, s_out.end_logits,
                    batch["start_positions"].to(device),
                    batch["end_positions"].to(device),
                ).item()

        avg_val = val_loss / len(val_loader)
        print(f"C组 Epoch {epoch+1} Val Loss: {avg_val:.4f}")
        best_val_loss = save_best(student, tokenizer, save_path, avg_val, best_val_loss, epoch)


# ============================================================
# D组：Logits + QA-Focused Attention 蒸馏（完整方案）
# ============================================================

def run_experiment_D():
    print("\n" + "="*60)
    print("D组：Logits + QA-Focused Attention 蒸馏（完整方案）")
    print("="*60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_SAVE_PATH)
    train_loader, val_loader = get_dataloaders(tokenizer, STUDENT_BATCH_SIZE)

    teacher = load_finetuned_teacher(device)
    student = build_student_model_from_teacher(device)
    optimizer, scheduler = make_optimizer_scheduler(
        student, train_loader, STUDENT_LR, STUDENT_WARMUP_RATIO, STUDENT_WEIGHT_DECAY, STUDENT_EPOCHS
    )

    save_path = EXPERIMENT_PATHS["D"]
    best_val_loss = float("inf")

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        total_loss_sum = hard_sum = logit_sum = attn_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[D组 Epoch {epoch+1}/{STUDENT_EPOCHS}] 训练")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            s_out = student(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            student_outputs = {
                "start_logits": s_out.start_logits,
                "end_logits": s_out.end_logits,
                "attentions": s_out.attentions,
            }
            teacher_outputs = {
                "start_logits": t_out.start_logits,
                "end_logits": t_out.end_logits,
                "attentions": t_out.attentions,
            }

            l_hard = hard_label_loss(s_out.start_logits, s_out.end_logits, start_pos, end_pos)
            l_kd = logits_distillation_loss(
                s_out.start_logits, s_out.end_logits,
                t_out.start_logits, t_out.end_logits,
                TEMPERATURE
            )
            l_attn = qa_focused_attention_loss(
                student_outputs["attentions"], teacher_outputs["attentions"],
                input_ids, token_type_ids, start_pos, end_pos
            )
            loss = ALPHA * l_hard + BETA * l_kd + GAMMA * l_attn

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss_sum += loss.item()
            hard_sum += l_hard.item()
            logit_sum += l_kd.item()
            attn_sum += l_attn.item()
            pbar.set_postfix(
                total=f"{loss.item():.4f}",
                hard=f"{l_hard.item():.4f}",
                kd=f"{l_kd.item():.4f}",
                attn=f"{l_attn.item():.4f}",
            )

        n = len(train_loader)
        print(
            f"D组 Epoch {epoch+1} Train | "
            f"Total: {total_loss_sum/n:.4f} | "
            f"Hard: {hard_sum/n:.4f} | "
            f"KD: {logit_sum/n:.4f} | "
            f"Attn: {attn_sum/n:.4f}"
        )

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s_out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )
                val_loss += hard_label_loss(
                    s_out.start_logits, s_out.end_logits,
                    batch["start_positions"].to(device),
                    batch["end_positions"].to(device),
                ).item()

        avg_val = val_loss / len(val_loader)
        print(f"D组 Epoch {epoch+1} Val Loss: {avg_val:.4f}")
        best_val_loss = save_best(student, tokenizer, save_path, avg_val, best_val_loss, epoch)


# ============================================================
# 汇总评估所有实验
# ============================================================

def run_evaluation(exp_list=None):
    """对所有已完成的实验 checkpoint 进行评估并输出对比报告"""
    if exp_list is None:
        exp_list = ["A", "B", "C", "D"]

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EXPERIMENT_NAMES = {
        "A": "A组-教师Baseline",
        "B": "B组-HardLabel蒸馏",
        "C": "C组-Logits蒸馏",
        "D": "D组-Logits+Attention蒸馏",
        "E": "E组-两阶段蒸馏(预训练+微调)",
    }

    for exp_id in exp_list:
        path = EXPERIMENT_PATHS[exp_id]
        if not os.path.exists(path):
            print(f"跳过 {exp_id} 组：checkpoint 不存在（{path}）")
            continue
        result = evaluate_model(
            model_path=path,
            experiment_name=EXPERIMENT_NAMES[exp_id],
            device=device,
        )
        all_results.append(result)

    if not all_results:
        print("没有可评估的实验结果")
        return

    # 保存 JSON
    os.makedirs("./results", exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至：{RESULTS_FILE}")

    # 打印对比表格
    print_comparison_table(all_results)
    return all_results


def print_comparison_table(results):
    """打印对比表格"""
    print("\n" + "="*90)
    print("实验对比报告")
    print("="*90)

    header = (
        f"{'实验':<28} {'EM%':>7} {'F1%':>7} "
        f"{'延迟(ms)':>10} {'P99(ms)':>9} "
        f"{'大小(MB)':>9} {'参数(M)':>8} "
        f"{'显存(MB)':>9}"
    )
    print(header)
    print("-"*90)

    for r in results:
        line = (
            f"{r['experiment']:<28} "
            f"{r['EM']:>7.2f} {r['F1']:>7.2f} "
            f"{r['avg_latency_ms']:>10.3f} {r['p99_latency_ms']:>9.3f} "
            f"{r['model_size_mb']:>9.1f} {r['param_count_M']:>8.1f} "
            f"{r['inference_gpu_mb']:>9.1f}"
        )
        print(line)

    print("="*90)

    # 相对于 A 组（教师 Baseline）计算压缩率和性能保留率
    baseline = next((r for r in results if "A组" in r["experiment"]), None)
    if baseline and len(results) > 1:
        print("\n相对教师 Baseline 的对比（学生模型各组）：")
        print(f"{'实验':<28} {'F1保留率':>10} {'大小压缩':>10} {'速度提升':>10} {'显存减少':>10}")
        print("-"*65)
        for r in results:
            if "A组" in r["experiment"]:
                continue
            f1_retention = r["F1"] / baseline["F1"] * 100 if baseline["F1"] > 0 else 0
            size_ratio = baseline["model_size_mb"] / r["model_size_mb"] if r["model_size_mb"] > 0 else 0
            speed_ratio = baseline["avg_latency_ms"] / r["avg_latency_ms"] if r["avg_latency_ms"] > 0 else 0
            mem_reduction = (1 - r["inference_gpu_mb"] / baseline["inference_gpu_mb"]) * 100 if baseline["inference_gpu_mb"] > 0 else 0
            print(
                f"{r['experiment']:<28} "
                f"{f1_retention:>9.1f}% "
                f"{size_ratio:>9.1f}x "
                f"{speed_ratio:>9.1f}x "
                f"{mem_reduction:>9.1f}%"
            )
        print("="*65)


# ============================================================
# 主入口
# ============================================================

def run_experiment_E():
    """
    E组：两阶段蒸馏
      第一阶段：预训练蒸馏（Wikipedia语料，MLM + 隐层 + Attention）
      第二阶段：微调蒸馏（CMRC2018，Hard Label + Logits + QA-Attention）
    学生初始化来自第一阶段产出，而非直接从微调教师偶数层拷贝
    """
    print("\n" + "="*60)
    print("E组：两阶段蒸馏（预训练蒸馏 -> 微调蒸馏）")
    print("="*60)

    # ---- 第一阶段：预训练蒸馏 ----
    if os.path.exists(PRETRAIN_STUDENT_SAVE_PATH):
        print(f"检测到预训练蒸馏权重已存在：{PRETRAIN_STUDENT_SAVE_PATH}，跳过第一阶段")
    else:
        print("\n>>> 第一阶段：预训练蒸馏")
        run_pretrain_distill()

    # ---- 第二阶段：微调蒸馏（与 D 组相同损失，但学生初始化不同）----
    print("\n>>> 第二阶段：微调阶段蒸馏（Logits + QA-Attention）")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_SAVE_PATH)
    train_loader, val_loader = get_dataloaders(tokenizer, STUDENT_BATCH_SIZE)

    teacher = load_finetuned_teacher(device)

    # ✅ 关键区别：从预训练蒸馏权重初始化，而非直接从微调教师偶数层
    student = build_student_model_from_teacher(device, init_mode="pretrain_distill")

    optimizer, scheduler = make_optimizer_scheduler(
        student, train_loader, STUDENT_LR, STUDENT_WARMUP_RATIO, STUDENT_WEIGHT_DECAY, STUDENT_EPOCHS
    )

    save_path = EXPERIMENT_PATHS["E"]
    best_val_loss = float("inf")

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        total_sum = hard_sum = logit_sum = attn_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[E组 Epoch {epoch+1}/{STUDENT_EPOCHS}] 微调蒸馏")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)

            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

            s_out = student(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

            l_hard = hard_label_loss(s_out.start_logits, s_out.end_logits, start_pos, end_pos)
            l_kd   = logits_distillation_loss(
                s_out.start_logits, s_out.end_logits,
                t_out["start_logits"], t_out["end_logits"],
                TEMPERATURE
            )
            l_attn = qa_focused_attention_loss(
                s_out.attentions, t_out["attentions"],
                input_ids, token_type_ids, start_pos, end_pos
            )
            loss = ALPHA * l_hard + BETA * l_kd + GAMMA * l_attn

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_sum += loss.item(); hard_sum += l_hard.item()
            logit_sum += l_kd.item(); attn_sum += l_attn.item()
            pbar.set_postfix(
                total=f"{loss.item():.4f}", hard=f"{l_hard.item():.4f}",
                kd=f"{l_kd.item():.4f}", attn=f"{l_attn.item():.4f}",
            )

        n = len(train_loader)
        print(
            f"E组 Epoch {epoch+1} | Total:{total_sum/n:.4f} | "
            f"Hard:{hard_sum/n:.4f} | KD:{logit_sum/n:.4f} | Attn:{attn_sum/n:.4f}"
        )

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s_out = student(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                )
                val_loss += hard_label_loss(
                    s_out.start_logits, s_out.end_logits,
                    batch["start_positions"].to(device),
                    batch["end_positions"].to(device),
                ).item()

        avg_val = val_loss / len(val_loader)
        print(f"E组 Epoch {epoch+1} Val Loss: {avg_val:.4f}")
        best_val_loss = save_best(student, tokenizer, save_path, avg_val, best_val_loss, epoch)


EXPERIMENT_RUNNERS = {
    "A": run_experiment_A,
    "B": run_experiment_B,
    "C": run_experiment_C,
    "D": run_experiment_D,
    "E": run_experiment_E,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        help="要运行的实验组：all / A / B / C / D / E / A,B,C 等")
    parser.add_argument("--report_only", action="store_true",
                        help="只汇总已有结果，不重新训练")
    args = parser.parse_args()

    if args.report_only:
        run_evaluation()
    else:
        if args.exp == "all":
            exp_list = ["A", "B", "C", "D", "E"]
        else:
            exp_list = [e.strip().upper() for e in args.exp.split(",")]

        for exp_id in exp_list:
            if exp_id not in EXPERIMENT_RUNNERS:
                print(f"未知实验组：{exp_id}，跳过")
                continue
            # B/C/D/E 依赖教师 checkpoint
            if exp_id in ("B", "C", "D", "E") and not os.path.exists(TEACHER_SAVE_PATH):
                print(f"[警告] 未找到微调教师 checkpoint，请先运行 A 组！")
                break
            EXPERIMENT_RUNNERS[exp_id]()

        run_evaluation(exp_list)
