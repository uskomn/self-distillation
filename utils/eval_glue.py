# glue_eval.py
"""
GLUE 评估模块：对预训练蒸馏后的学生模型在 GLUE 8个任务上微调并评估

任务列表及指标：
  CoLA  : Matthews 相关系数（MCC）
  SST-2 : 准确率（Acc）
  MRPC  : F1 + 准确率
  STS-B : Pearson / Spearman 相关系数（回归任务）
  QQP   : F1 + 准确率
  MNLI  : 匹配/不匹配准确率
  QNLI  : 准确率
  RTE   : 准确率

用法：
  # 评估预训练蒸馏后的学生
  python glue_eval.py

  # 同时评估教师（bert-base-uncased）作对比基准
  python glue_eval.py --also_eval_teacher

  # 评估指定 checkpoint
  python glue_eval.py --model_path ./checkpoints_squad/student_pretrain_distilled
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    BertForMaskedLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm

from config_squad import (
    PT_TEACHER_NAME,
    PT_STUDENT_SAVE_PATH,
    STUDENT_BASE,
    STUDENT_NUM_LAYERS,
    GLUE_TASKS,
    GLUE_EPOCHS,
    GLUE_BATCH_SIZE,
    GLUE_LR,
    GLUE_MAX_LENGTH,
    GLUE_WARMUP_RATIO,
    GLUE_WEIGHT_DECAY,
    GLUE_SAVE_DIR,
    GLUE_RESULTS_FILE,
)


# ============================================================
# 任务元信息
# ============================================================

TASK_CONFIG = {
    "cola": {"keys": ("sentence",  None),           "num_labels": 2, "metric": "mcc"},
    "sst2": {"keys": ("sentence",  None),           "num_labels": 2, "metric": "acc"},
    "mrpc": {"keys": ("sentence1", "sentence2"),    "num_labels": 2, "metric": "f1_acc"},
    "stsb": {"keys": ("sentence1", "sentence2"),    "num_labels": 1, "metric": "pearson_spearman"},
    "qqp":  {"keys": ("question1", "question2"),    "num_labels": 2, "metric": "f1_acc"},
    "mnli": {"keys": ("premise",   "hypothesis"),   "num_labels": 3, "metric": "acc"},
    "qnli": {"keys": ("question",  "sentence"),     "num_labels": 2, "metric": "acc"},
    "rte":  {"keys": ("sentence1", "sentence2"),    "num_labels": 2, "metric": "acc"},
}


# ============================================================
# 数据加载
# ============================================================

def get_glue_dataloader(task, tokenizer, split="train"):
    cfg       = TASK_CONFIG[task]
    key1, key2 = cfg["keys"]

    # MNLI 验证有 matched / mismatched 两份
    actual_split = split
    if task == "mnli" and split == "validation":
        actual_split = "validation_matched"

    ds = load_from_disk(f"/root/autodl-tmp/glue_disk/{task}")
    ds = ds[actual_split]

    def tokenize_fn(examples):
        if key2 is None:
            return tokenizer(
                examples[key1],
                max_length=GLUE_MAX_LENGTH, truncation=True, padding="max_length",
            )
        return tokenizer(
            examples[key1], examples[key2],
            max_length=GLUE_MAX_LENGTH, truncation=True, padding="max_length",
        )

    tokenized = ds.map(tokenize_fn, batched=True, desc=f"Tokenizing {task}/{actual_split}")
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    return DataLoader(tokenized, batch_size=GLUE_BATCH_SIZE, shuffle=(split == "train"))


# ============================================================
# 指标计算
# ============================================================

def compute_metrics(task, preds, labels):
    metric = TASK_CONFIG[task]["metric"]
    if metric == "mcc":
        return {"MCC": round(float(matthews_corrcoef(labels, preds)), 4)}
    elif metric == "acc":
        acc = (np.array(preds) == np.array(labels)).mean()
        return {"Acc": round(float(acc), 4)}
    elif metric == "f1_acc":
        f1  = f1_score(labels, preds, average="binary")
        acc = (np.array(preds) == np.array(labels)).mean()
        return {"F1": round(float(f1), 4), "Acc": round(float(acc), 4)}
    elif metric == "pearson_spearman":
        pr, _ = pearsonr(preds, labels)
        sr, _ = spearmanr(preds, labels)
        return {"Pearson": round(float(pr), 4), "Spearman": round(float(sr), 4)}
    return {}


# ============================================================
# 模型构建
# ============================================================

def build_student_for_glue(task, model_path):
    """
    从预训练蒸馏的 BertForMaskedLM 权重取出 bert 子模块，
    接上对应 GLUE 任务的分类/回归 head
    """
    num_labels = TASK_CONFIG[task]["num_labels"]
    config = BertConfig.from_pretrained(STUDENT_BASE)
    config.num_hidden_layers = STUDENT_NUM_LAYERS
    config.num_labels = num_labels

    model = BertForSequenceClassification(config)

    # 加载预训练蒸馏权重（BertForMaskedLM），取 bert 子模块
    pretrained = BertForMaskedLM.from_pretrained(model_path)
    model.bert.load_state_dict(pretrained.bert.state_dict(),strict=False)
    del pretrained
    torch.cuda.empty_cache()
    return model


def build_teacher_for_glue(task):
    """构建教师（bert-base-uncased 12层）用于对比基准"""
    num_labels = TASK_CONFIG[task]["num_labels"]
    return AutoModelForSequenceClassification.from_pretrained(
        PT_TEACHER_NAME, num_labels=num_labels
    )


# ============================================================
# 单任务微调 + 评估
# ============================================================

def finetune_and_eval(task, model, tokenizer, device, save_path=None):
    train_loader = get_glue_dataloader(task, tokenizer, split="train")
    val_loader   = get_glue_dataloader(task, tokenizer, split="validation")

    model = model.to(device)
    is_regression = (TASK_CONFIG[task]["num_labels"] == 1)

    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=GLUE_LR, weight_decay=GLUE_WEIGHT_DECAY
    )
    total_steps  = len(train_loader) * GLUE_EPOCHS
    warmup_steps = int(total_steps * GLUE_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score   = float("-inf")
    best_metrics = {}

    for epoch in range(GLUE_EPOCHS):
        # 训练
        model.train()
        for batch in tqdm(train_loader,
                          desc=f"  [{task.upper()} Epoch {epoch+1}/{GLUE_EPOCHS}]",
                          leave=False):
            labels = batch["labels"].to(device)
            if is_regression:
                labels = labels.float()

            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=labels,
            )
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                ).logits.cpu()

                if is_regression:
                    all_preds.extend(logits.squeeze(-1).tolist())
                    all_labels.extend(batch["labels"].float().tolist())
                else:
                    all_preds.extend(logits.argmax(dim=-1).tolist())
                    all_labels.extend(batch["labels"].tolist())

        metrics = compute_metrics(task, all_preds, all_labels)
        score   = list(metrics.values())[0]
        print(f"  {task.upper()} Epoch {epoch+1} | {metrics}")

        if score > best_score:
            best_score   = score
            best_metrics = metrics.copy()
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)

    # MNLI 额外评估 mismatched
    if task == "mnli":
        val_mm = get_glue_dataloader(task, tokenizer, split="validation_mismatched")
        model.eval()
        preds_mm, labels_mm = [], []
        with torch.no_grad():
            for batch in val_mm:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                ).logits.cpu()
                preds_mm.extend(logits.argmax(dim=-1).tolist())
                labels_mm.extend(batch["labels"].tolist())
        acc_mm = (np.array(preds_mm) == np.array(labels_mm)).mean()
        best_metrics["Acc_mm"] = round(float(acc_mm), 4)

    return best_metrics


# ============================================================
# 主评估流程
# ============================================================

def run_glue_eval(model_path, model_label="学生（预训练蒸馏）", also_eval_teacher=False):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PT_TEACHER_NAME)
    student_results = {}
    teacher_results = {}

    print("\n" + "=" * 65)
    print(f"GLUE 评估：{model_label}")
    print(f"  权重路径：{model_path}")
    print(f"  任务列表：{[t.upper() for t in GLUE_TASKS]}")
    print("=" * 65)

    # for task in GLUE_TASKS:
    #     print(f"\n▶ {task.upper()}")
    #     model     = build_student_for_glue(task, model_path)
    #     save_path = os.path.join(GLUE_SAVE_DIR, f"student_{task}")
    #     metrics   = finetune_and_eval(task, model, tokenizer, device, save_path)
    #     student_results[task] = metrics
    #     print(f"   最优：{metrics}")
    #     del model
    #     torch.cuda.empty_cache()

    if also_eval_teacher:
        print("\n" + "=" * 65)
        print("GLUE 评估：教师基线（bert-base-uncased 12层）")
        print("=" * 65)
        for task in GLUE_TASKS:
            print(f"\n▶ {task.upper()}（教师）")
            model     = build_teacher_for_glue(task)
            save_path = os.path.join(GLUE_SAVE_DIR, f"teacher_{task}")
            metrics   = finetune_and_eval(task, model, tokenizer, device, save_path)
            teacher_results[task] = metrics
            print(f"  教师最优：{metrics}")
            del model
            torch.cuda.empty_cache()

    _print_glue_table(student_results, teacher_results)

    os.makedirs(os.path.dirname(GLUE_RESULTS_FILE), exist_ok=True)
    with open(GLUE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"student": student_results,
             "teacher": teacher_results or "未评估"},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nGLUE 结果已保存至：{GLUE_RESULTS_FILE}")
    return student_results


def _print_glue_table(student_results, teacher_results=None):
    print("\n" + "=" * 75)
    print("GLUE 汇总")
    print(f"{'任务':<8} {'指标':<12} {'学生（蒸馏）':>14}", end="")
    if teacher_results:
        print(f" {'教师（base）':>14} {'保留率':>8}", end="")
    print()
    print("-" * 75)

    for task in GLUE_TASKS:
        sm = student_results.get(task, {})
        tm = teacher_results.get(task, {}) if teacher_results else {}
        for i, (key, sval) in enumerate(sm.items()):
            tval = tm.get(key, None)
            row  = f"{task.upper() if i==0 else '':<8} {key:<12} {sval:>14.4f}"
            if teacher_results:
                if tval is not None:
                    retain = sval / tval * 100 if tval != 0 else float("nan")
                    row += f" {tval:>14.4f} {retain:>7.1f}%"
                else:
                    row += f" {'N/A':>14}"
            print(row)

    print("=" * 75)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default=PT_STUDENT_SAVE_PATH,
        help="要评估的模型路径（默认：预训练蒸馏学生）",
    )
    parser.add_argument(
        "--also_eval_teacher", action="store_true",
        help="同时评估教师 bert-base-uncased 作对比基准",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"模型路径不存在：{args.model_path}")
        print("请先运行预训练蒸馏：python pretrain_distill_squad.py")
        exit(1)

    run_glue_eval(
        model_path=args.model_path,
        model_label="学生（预训练蒸馏后）",
        also_eval_teacher=args.also_eval_teacher,
    )
    os.system("/usr/bin/shutdown")