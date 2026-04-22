# ablation_squad.py
"""
消融实验：SQuAD 两阶段蒸馏

实验组设计：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【预训练蒸馏方式消融】（3组，微调阶段损失相同）
  A1 - No-PT      : 无预训练蒸馏，直接微调蒸馏（基线）
  A2 - PT-Single  : 普通预训练蒸馏（单教师 bert-base，仅 Wiki，无 QA 注入）
  A3 - PT-Dual    : 双教师任务感知预训练蒸馏（Wiki + SQuAD QA 混入）

【损失项消融】（4组，均基于 PT-Dual 预训练权重）
  B1 - Full       : MLM + Logits + Hidden + Attn（完整四项）
  B2 - No-Logits  : MLM + Hidden + Attn（去掉 L_logits）
  B3 - No-Hidden  : MLM + Logits + Attn（去掉 L_hidden）
  B4 - No-Attn    : MLM + Logits + Hidden（去掉 L_attn）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法：
  python ablation_squad.py --exp all            # 跑全部7组
  python ablation_squad.py --exp A1,A2,A3       # 只跑预训练消融
  python ablation_squad.py --exp B1,B2,B3,B4   # 只跑损失消融
  python ablation_squad.py --report_only        # 只汇总已有结果
"""

import os
import json
import argparse
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
    TEACHER_SAVE_PATH, STUDENT_BASE,
    DATASET_NAME, MAX_LENGTH, DOC_STRIDE, MAX_ANSWER_LENGTH,
    STUDENT_NUM_LAYERS, TEACHER_LAYERS_FOR_DISTILL,
    EPOCHS, BATCH_SIZE, LR, WARMUP_RATIO, WEIGHT_DECAY,
    ALPHA, BETA, GAMMA, TEMPERATURE,
    PT_STUDENT_SAVE_PATH,
    PT_TEACHER_LAYERS,PT_TEACHER_NAME,
    PT_LAMBDA_MLM, PT_LAMBDA_LOGITS, PT_LAMBDA_HIDDEN, PT_LAMBDA_ATTN,
    PT_QA_INTERLEAVE_EVERY, PT_LAMBDA_QA_HARD, PT_LAMBDA_QA_KD, PT_USE_QA_KD,
)
from models.models_squad import TeacherModel, StudentModel
from train_squad_distill import (
    preprocess_train, preprocess_validation,
    hard_label_loss, logits_distillation_loss, qa_focused_attention_loss,
    postprocess_predictions, compute_exact, compute_f1,
)
from utils.pretrain_distill_squad import (
    PretrainTeacher, PretrainStudent, QATeacher, QAHead,
    wiki_distill_loss, qa_task_loss,
    get_wiki_dataloader, get_qa_dataloader,
)


# ============================================================
# 消融实验路径配置
# ============================================================

ABLATION_SAVE_DIR     = "/root/autodl-tmp/checkpoints_squad/ablation"
ABLATION_RESULTS_FILE = "/root/autodl-tmp/results_squad/ablation_results.json"

# 预训练蒸馏权重路径（各实验组独立保存）
PT_PATHS = {
    "A2": os.path.join(ABLATION_SAVE_DIR, "pt_single"),   # 单教师普通预训练蒸馏
    "A3": PT_STUDENT_SAVE_PATH,                            # 双教师任务感知蒸馏（复用主实验）
    # 损失消融组全部基于 A3（PT-Dual）的预训练权重，只重跑微调阶段
    "B1": PT_STUDENT_SAVE_PATH,
    "B2": PT_STUDENT_SAVE_PATH,
    "B3": PT_STUDENT_SAVE_PATH,
    "B4": PT_STUDENT_SAVE_PATH,
}

# 微调阶段学生 checkpoint 路径
FT_PATHS = {
    "A1": os.path.join(ABLATION_SAVE_DIR, "ft_A1_no_pt"),
    "A2": os.path.join(ABLATION_SAVE_DIR, "ft_A2_pt_single"),
    "A3": os.path.join(ABLATION_SAVE_DIR, "ft_A3_pt_dual"),
    "B1": os.path.join(ABLATION_SAVE_DIR, "ft_B1_full"),
    "B2": os.path.join(ABLATION_SAVE_DIR, "ft_B2_no_logits"),
    "B3": os.path.join(ABLATION_SAVE_DIR, "ft_B3_no_hidden"),
    "B4": os.path.join(ABLATION_SAVE_DIR, "ft_B4_no_attn"),
}

EXP_NAMES = {
    "A1": "A1-No-PT（无预训练蒸馏）",
    "A2": "A2-PT-Single（单教师Wiki）",
    "A3": "A3-PT-Dual（双教师+QA注入）",
    "B1": "B1-Full（MLM+Logits+Hidden+Attn）",
    "B2": "B2-No-Logits（去掉L_logits）",
    "B3": "B3-No-Hidden（去掉L_hidden）",
    "B4": "B4-No-Attn（去掉L_attn）",
}


# ============================================================
# 数据工具
# ============================================================

def get_squad_dataloaders(tokenizer):
    dataset = load_from_disk(DATASET_NAME)

    train_ds = dataset["train"].map(
        lambda x: preprocess_train(x, tokenizer),
        batched=True, remove_columns=dataset["train"].column_names,
        desc="处理训练集",
    )
    val_ds = dataset["validation"].map(
        lambda x: preprocess_validation(x, tokenizer),
        batched=True, remove_columns=dataset["validation"].column_names,
        desc="处理验证集",
    )

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
    return train_loader, val_loader, val_raw, val_meta


# ============================================================
# 微调阶段评估
# ============================================================

def evaluate_squad(model_wrapper, val_loader, val_raw, val_meta, device):
    model_wrapper.eval() if hasattr(model_wrapper, "eval") else None
    model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper
    model.eval()

    all_start, all_end = [], []
    with torch.no_grad():
        for batch in val_loader:
            out = model_wrapper(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            all_start.append(out["start_logits"].cpu().numpy())
            all_end.append(out["end_logits"].cpu().numpy())

    all_start = np.concatenate(all_start, axis=0)
    all_end   = np.concatenate(all_end,   axis=0)
    preds = postprocess_predictions(val_raw, val_meta, all_start, all_end)

    em_scores, f1_scores = [], []
    for ex in val_raw:
        pred  = preds.get(ex["id"], "")
        golds = ex["answers"]["text"]
        em_scores.append(max(compute_exact(pred, g) for g in golds))
        f1_scores.append(max(compute_f1(pred, g) for g in golds))

    return {
        "EM": round(np.mean(em_scores) * 100, 2),
        "F1": round(np.mean(f1_scores) * 100, 2),
    }


# ============================================================
# 微调阶段蒸馏训练（公共函数）
# ============================================================

def finetune_distill(student_init_mode, save_path, device,
                     train_loader, val_loader, val_raw, val_meta, tokenizer,
                     pretrain_path=None):
    """
    微调阶段蒸馏训练，所有消融组共用
    pretrain_path：预训练蒸馏权重路径，None 时使用 config 默认路径
    """
    teacher = TeacherModel(load_finetuned=True).to(device)
    teacher.eval()

    student_wrapper = StudentModel(init_mode=student_init_mode, pretrain_path=pretrain_path)
    student = student_wrapper.model.to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    best_f1    = 0.0
    best_metrics = {}

    for epoch in range(EPOCHS):
        student.train()
        for batch in tqdm(train_loader, desc=f"  [微调 Epoch {epoch+1}/{EPOCHS}]", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)

            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            s_out = student_wrapper(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

            l_hard   = hard_label_loss(s_out["start_logits"], s_out["end_logits"], start_pos, end_pos)
            l_logits = logits_distillation_loss(
                s_out["start_logits"], s_out["end_logits"],
                t_out["start_logits"], t_out["end_logits"],
            )
            l_attn = qa_focused_attention_loss(
                s_out["attentions"], t_out["attentions"],
                input_ids, token_type_ids, start_pos, end_pos,
            )
            loss = ALPHA * l_hard + BETA * l_logits + GAMMA * l_attn

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        metrics = evaluate_squad(student_wrapper, val_loader, val_raw, val_meta, device)
        print(f"  Epoch {epoch+1} | EM:{metrics['EM']:.2f}% F1:{metrics['F1']:.2f}%")

        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_metrics = metrics.copy()
            os.makedirs(save_path, exist_ok=True)
            student.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    return best_metrics


# ============================================================
# 单教师普通预训练蒸馏（A2：只用 Wiki，无 QA 注入，无 QA 教师）
# ============================================================

def run_pt_single(device, tokenizer):
    """
    A2 单教师预训练蒸馏：
    - 教师：bert-base-uncased（原始预训练，冻结）
    - 数据：仅英文 Wikipedia，无 SQuAD QA 注入
    - 损失：MLM + Logits + Hidden + Attn（与 PT-Dual 相同，去掉 QA 部分）
    """
    from itertools import cycle
    from utils.pretrain_distill_squad import _load_raw_wiki
    from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
    from config_squad import PT_TEACHER_NAME, PT_EPOCHS, PT_BATCH_SIZE, PT_LR, PT_WARMUP_RATIO, PT_WEIGHT_DECAY, PT_MLM_PROB

    print("=" * 55)
    print("A2 - PT-Single：单教师普通预训练蒸馏（仅 Wiki）")
    print("=" * 55)

    save_path = PT_PATHS["A2"]
    if os.path.exists(save_path):
        print(f"已有 checkpoint，跳过：{save_path}")
        return

    pretrain_teacher = PretrainTeacher().to(device)
    pretrain_teacher.eval()
    # A2 无 QA 教师

    student_wrapper = PretrainStudent()
    student = student_wrapper.model.to(device)

    # 仅 Wiki 数据，无 QA
    wiki_loader = get_wiki_dataloader(tokenizer)

    optimizer   = torch.optim.AdamW(student.parameters(), lr=PT_LR, weight_decay=PT_WEIGHT_DECAY)
    total_steps = len(wiki_loader) * PT_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * PT_WARMUP_RATIO), total_steps
    )

    best_loss = float("inf")
    for epoch in range(PT_EPOCHS):
        student.train()
        total_sum = 0.0

        pbar = tqdm(wiki_loader, desc=f"[A2-PT-Single Epoch {epoch+1}/{PT_EPOCHS}]")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            t_hidden, t_attn, t_logits = pretrain_teacher(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            s_out = student_wrapper(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, labels=labels,
            )

            #  与 PT-Dual 相同的 Wiki 损失（无 QA 项）
            loss, l_mlm, l_logits, l_hidden, l_attn = wiki_distill_loss(
                s_out, t_hidden, t_attn, t_logits, attention_mask=attention_mask,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_sum / len(wiki_loader)
        print(f"A2 Epoch {epoch+1} | Loss:{avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(save_path, exist_ok=True)
            student.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  保存 A2 权重 -> {save_path}")


# ============================================================
# 损失消融：预训练蒸馏（基于不同损失组合）
# ============================================================

def run_pt_ablation_loss(exp_id, device, tokenizer):
    """
    B1-B4 损失消融：重跑预训练蒸馏，每次去掉一项损失
    B1-Full      : λ_logits=0.5, λ_hidden=0.1, λ_attn=0.1
    B2-No-Logits : λ_logits=0.0
    B3-No-Hidden : λ_hidden=0.0
    B4-No-Attn   : λ_attn=0.0
    """
    loss_configs = {
        "B1": {"lambda_logits": PT_LAMBDA_LOGITS, "lambda_hidden": PT_LAMBDA_HIDDEN,
               "lambda_attn": PT_LAMBDA_ATTN,   "desc": "Full（MLM+Logits+Hidden+Attn）"},
        "B2": {"lambda_logits": 0.0,             "lambda_hidden": PT_LAMBDA_HIDDEN,
               "lambda_attn": PT_LAMBDA_ATTN,   "desc": "No-Logits（去掉L_logits）"},
        "B3": {"lambda_logits": PT_LAMBDA_LOGITS, "lambda_hidden": 0.0,
               "lambda_attn": PT_LAMBDA_ATTN,   "desc": "No-Hidden（去掉L_hidden）"},
        "B4": {"lambda_logits": PT_LAMBDA_LOGITS, "lambda_hidden": PT_LAMBDA_HIDDEN,
               "lambda_attn": 0.0,              "desc": "No-Attn（去掉L_attn）"},
    }
    cfg      = loss_configs[exp_id]
    save_path = os.path.join(ABLATION_SAVE_DIR, f"pt_{exp_id.lower()}")

    # B1 直接复用 PT-Dual 的权重（完整损失配置，已在主实验跑过）
    if exp_id == "B1":
        if os.path.exists(PT_STUDENT_SAVE_PATH):
            print(f"B1-Full 复用 PT-Dual 权重：{PT_STUDENT_SAVE_PATH}")
            PT_PATHS["B1"] = PT_STUDENT_SAVE_PATH
            return   # 路径已更新，直接返回，finetune 阶段会用 PT_PATHS["B1"]
        # PT-Dual 未跑则重新跑，保存到 PT_STUDENT_SAVE_PATH
        save_path = PT_STUDENT_SAVE_PATH

    if os.path.exists(save_path):
        print(f"已有 {exp_id} 预训练权重，跳过：{save_path}")
        PT_PATHS[exp_id] = save_path
        return

    print(f"\n{'='*55}")
    print(f"{exp_id} - 预训练蒸馏消融：{cfg['desc']}")
    print(f"  λ_logits={cfg['lambda_logits']}  λ_hidden={cfg['lambda_hidden']}  λ_attn={cfg['lambda_attn']}")
    print(f"{'='*55}")

    from itertools import cycle
    from config_squad import PT_EPOCHS, PT_LR, PT_WARMUP_RATIO, PT_WEIGHT_DECAY

    pretrain_teacher = PretrainTeacher().to(device)
    pretrain_teacher.eval()

    # B 组使用双教师（与 PT-Dual 相同，只改损失权重）
    qa_teacher = None
    if PT_USE_QA_KD and os.path.exists(TEACHER_SAVE_PATH):
        qa_teacher = QATeacher().to(device)
        qa_teacher.eval()

    student_wrapper = PretrainStudent()
    student = student_wrapper.model.to(device)
    qa_head = QAHead(student.config.hidden_size).to(device)

    wiki_loader = get_wiki_dataloader(tokenizer)
    qa_loader   = get_qa_dataloader(tokenizer)
    qa_iter     = cycle(qa_loader)

    all_params  = list(student.parameters()) + list(qa_head.parameters())
    optimizer   = torch.optim.AdamW(all_params, lr=PT_LR, weight_decay=PT_WEIGHT_DECAY)
    total_steps = len(wiki_loader) * PT_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * PT_WARMUP_RATIO), total_steps
    )

    best_loss = float("inf")
    for epoch in range(PT_EPOCHS):
        student.train(); qa_head.train()
        total_sum = 0.0

        pbar = tqdm(wiki_loader, desc=f"[{exp_id} Epoch {epoch+1}/{PT_EPOCHS}]")
        for wiki_step, wiki_batch in enumerate(pbar):
            input_ids      = wiki_batch["input_ids"].to(device)
            attention_mask = wiki_batch["attention_mask"].to(device)
            labels         = wiki_batch["labels"].to(device)
            token_type_ids = wiki_batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            t_hidden, t_attn, t_logits = pretrain_teacher(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )
            s_out = student_wrapper(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, labels=labels,
            )

            # 按消融配置设置对应 lambda
            loss, l_mlm, l_logits_v, l_hidden_v, l_attn_v = wiki_distill_loss(
                s_out, t_hidden, t_attn, t_logits,
                attention_mask=attention_mask,
                lambda_logits=cfg["lambda_logits"],
                lambda_hidden=cfg["lambda_hidden"],
                lambda_attn=cfg["lambda_attn"],
            )

            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_sum += loss.item()

            # QA batch 插入
            if (wiki_step + 1) % PT_QA_INTERLEAVE_EVERY == 0:
                qa_batch        = next(qa_iter)
                qa_input_ids    = qa_batch["input_ids"].to(device)
                qa_attn_mask    = qa_batch["attention_mask"].to(device)
                qa_tt_ids       = qa_batch["token_type_ids"].to(device)
                start_pos       = qa_batch["start_positions"].to(device)
                end_pos         = qa_batch["end_positions"].to(device)

                bert_out        = student_wrapper.get_bert_output(qa_input_ids, qa_attn_mask, qa_tt_ids)
                s_start, s_end  = qa_head(bert_out.last_hidden_state)
                t_start = t_end = None
                if qa_teacher is not None:
                    t_start, t_end = qa_teacher(qa_input_ids, qa_attn_mask, qa_tt_ids)

                loss_qa, _, _ = qa_task_loss(s_start, s_end, start_pos, end_pos, t_start, t_end)
                loss_qa.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step(); optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_sum / len(wiki_loader)
        print(f"{exp_id} Epoch {epoch+1} | Loss:{avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            os.makedirs(save_path, exist_ok=True)
            student.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f" 保存 {exp_id} 预训练权重 -> {save_path}")

    PT_PATHS[exp_id] = save_path


# ============================================================
# 各实验组运行入口
# ============================================================

def run_experiment(exp_id, device, train_loader, val_loader, val_raw, val_meta, tokenizer):
    print(f"\n{'='*60}")
    print(f"实验组：{EXP_NAMES[exp_id]}")
    print(f"{'='*60}")

    # ------ 确定预训练初始化方式和权重路径 ------
    student_init  = "bert_base"   # A1 默认
    pretrain_path = None

    if exp_id == "A1":
        # 无预训练蒸馏，直接用 bert_base 初始化，无需路径
        student_init  = "bert_base"
        pretrain_path = None

    elif exp_id == "A2":
        # 先跑单教师预训练蒸馏，保存到 PT_PATHS["A2"]
        run_pt_single(device, tokenizer)
        student_init  = "pretrain_distill"
        pretrain_path = PT_PATHS["A2"]   #  显式传入 A2 专属路径

    elif exp_id == "A3":
        # 复用主实验的 PT-Dual 权重
        if not os.path.exists(PT_STUDENT_SAVE_PATH):
            print(f"未找到 PT-Dual 权重：{PT_STUDENT_SAVE_PATH}")
            print("请先运行：python pretrain_distill_squad.py")
            return None
        student_init  = "pretrain_distill"
        pretrain_path = PT_STUDENT_SAVE_PATH

    elif exp_id in ("B1", "B2", "B3", "B4"):
        # 损失消融：先跑对应消融配置的预训练蒸馏
        run_pt_ablation_loss(exp_id, device, tokenizer)
        student_init  = "pretrain_distill"
        pretrain_path = PT_PATHS[exp_id]  # 每组用自己的预训练权重

    # ------ 微调蒸馏阶段 ------
    save_path = FT_PATHS[exp_id]
    print(exp_id)
    print(pretrain_path)
    metrics   = finetune_distill(
        student_init_mode=student_init,
        save_path=save_path,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        val_raw=val_raw,
        val_meta=val_meta,
        tokenizer=tokenizer,
        pretrain_path=pretrain_path,     # 显式传入，不再修改全局变量
    )

    print(f" {EXP_NAMES[exp_id]} 完成 | EM:{metrics['EM']:.2f}% F1:{metrics['F1']:.2f}%")
    return metrics


# ============================================================
# 结果汇总打印
# ============================================================

def print_ablation_table(results: dict):
    print("\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)

    sections = [
        ("【预训练蒸馏方式消融】", ["A1", "A2", "A3"]),
        ("【损失项消融（均基于 PT-Dual）】", ["B1", "B2", "B3", "B4"]),
    ]

    for title, exp_ids in sections:
        print(f"\n{title}")
        print(f"  {'实验组':<38} {'EM%':>7} {'F1%':>7}  {'说明'}")
        print("  " + "-" * 65)
        baseline_f1 = None
        for exp_id in exp_ids:
            if exp_id not in results:
                print(f"  {EXP_NAMES[exp_id]:<38} {'N/A':>7} {'N/A':>7}")
                continue
            m = results[exp_id]
            if baseline_f1 is None:
                baseline_f1 = m["F1"]
                delta = ""
            else:
                diff = m["F1"] - baseline_f1
                delta = f"  ({'+' if diff >= 0 else ''}{diff:.2f})"
            print(f"  {EXP_NAMES[exp_id]:<38} {m['EM']:>7.2f} {m['F1']:>7.2f}{delta}")

    print("\n" + "=" * 70)
    print("注：损失消融 delta 相对于 B1-Full 基线")


# ============================================================
# 主入口
# ============================================================

ALL_EXPS = ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]

if __name__ == "__main__":
    import os

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp", type=str, default="all",
                            help="实验组：all / A1,A2,A3 / B1,B2,B3,B4 等")
        parser.add_argument("--report_only", action="store_true",
                            help="只汇总已有结果，不重新训练")
        args = parser.parse_args()

        # 加载已有结果
        existing_results = {}
        if os.path.exists(ABLATION_RESULTS_FILE):
            with open(ABLATION_RESULTS_FILE, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

        if args.report_only:
            print_ablation_table(existing_results)
        else:
            exp_list = ALL_EXPS if args.exp == "all" else [e.strip().upper() for e in args.exp.split(",")]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"设备：{device}")
            if torch.cuda.is_available():
                print(f"GPU：{torch.cuda.get_device_name(0)}")

            tokenizer = AutoTokenizer.from_pretrained(PT_TEACHER_NAME)
            train_loader, val_loader, val_raw, val_meta = get_squad_dataloaders(tokenizer)

            # 检查前置条件
            if any(e in exp_list for e in ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]):
                if not os.path.exists(TEACHER_SAVE_PATH):
                    print(f"[错误] 未找到微调教师 checkpoint：{TEACHER_SAVE_PATH}")
                    print("请先运行：python train_squad_distill.py --teacher_only")
                    exit(1)

            all_results = dict(existing_results)

            for exp_id in exp_list:
                if exp_id not in ALL_EXPS:
                    print(f"未知实验组：{exp_id}，跳过")
                    continue

                metrics = run_experiment(
                    exp_id, device, train_loader, val_loader, val_raw, val_meta, tokenizer
                )

                if metrics is not None:
                    all_results[exp_id] = metrics
                    os.makedirs(os.path.dirname(ABLATION_RESULTS_FILE), exist_ok=True)
                    with open(ABLATION_RESULTS_FILE, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)

            print_ablation_table(all_results)
            print(f"\n消融实验结果已保存至：{ABLATION_RESULTS_FILE}")

    finally:
        print("程序结束（无论成功或失败），准备关机...")
        os.system("/usr/bin/shutdown")