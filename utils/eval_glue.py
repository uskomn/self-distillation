# eval_glue.py
"""
GLUE 基准测试：评估预训练蒸馏后学生模型的迁移能力

支持任务：
  分类类：SST-2, MNLI, QNLI, QQP, RTE, MRPC, CoLA, WNLI
  回归类：STS-B

流程（每个任务独立）：
  1. 在任务训练集上 fine-tune 学生模型（轻量级，少 epoch）
  2. 在验证集上评估指标
  3. 汇总打印所有任务分数

用法：
  python eval_glue.py                          # 测所有任务
  python eval_glue.py --tasks sst2 mnli        # 只测指定任务
  python eval_glue.py --model_path /your/path  # 指定模型路径
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from config import PRETRAIN_STUDENT_SAVE_PATH, STUDENT_NUM_LAYERS, MAX_LENGTH


# ============================================================
# GLUE 任务配置
# ============================================================

GLUE_TASKS = {
    "cola": {
        "dataset": ("glue", "cola"),
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "matthews",
        "split_eval": "validation",
    },
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "accuracy",
        "split_eval": "validation",
    },
    "mrpc": {
        "dataset": ("glue", "mrpc"),
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "f1_accuracy",   # MRPC 官方：F1 + Accuracy
        "split_eval": "validation",
    },
    "stsb": {
        "dataset": ("glue", "stsb"),
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 1,           # 回归
        "metric": "pearson_spearman",
        "split_eval": "validation",
    },
    "qqp": {
        "dataset": ("glue", "qqp"),
        "text_keys": ("question1", "question2"),
        "num_labels": 2,
        "metric": "f1_accuracy",
        "split_eval": "validation",
    },
    "mnli": {
        "dataset": ("glue", "mnli"),
        "text_keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "metric": "accuracy",
        "split_eval": "validation_matched",
    },
    "qnli": {
        "dataset": ("glue", "qnli"),
        "text_keys": ("question", "sentence"),
        "num_labels": 2,
        "metric": "accuracy",
        "split_eval": "validation",
    },
    "rte": {
        "dataset": ("glue", "rte"),
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
        "split_eval": "validation",
    },
    "wnli": {
        "dataset": ("glue", "wnli"),
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
        "split_eval": "validation",
    },
}

# fine-tune 超参（轻量级，用于评估而非刷榜）
FINETUNE_CONFIG = {
    "default": {"epochs": 3, "batch_size": 32, "lr": 2e-5, "warmup_ratio": 0.1},
    # 小数据集多训几轮
    "rte":  {"epochs": 5, "batch_size": 16, "lr": 3e-5, "warmup_ratio": 0.1},
    "mrpc": {"epochs": 5, "batch_size": 16, "lr": 3e-5, "warmup_ratio": 0.1},
    "cola": {"epochs": 5, "batch_size": 16, "lr": 2e-5, "warmup_ratio": 0.1},
    "wnli": {"epochs": 5, "batch_size": 16, "lr": 2e-5, "warmup_ratio": 0.1},
    "stsb": {"epochs": 5, "batch_size": 16, "lr": 2e-5, "warmup_ratio": 0.1},
}


# ============================================================
# 数据预处理
# ============================================================

def preprocess_glue(task_name, tokenizer, max_length=MAX_LENGTH):
    cfg = GLUE_TASKS[task_name]
    ds_name, ds_config = cfg["dataset"]

    print(f"  加载数据集 {ds_name}/{ds_config} ...")
    try:
        dataset = load_dataset(ds_name, ds_config, trust_remote_code=True)
    except Exception as e:
        print(f"  下载失败（{e}），跳过 {task_name}")
        return None, None

    k1, k2 = cfg["text_keys"]

    def tokenize_fn(examples):
        if k2 is None:
            return tokenizer(
                examples[k1],
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
        return tokenizer(
            examples[k1],
            examples[k2],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

    remove_cols = [c for c in dataset["train"].column_names if c != "label"]
    tokenized = dataset.map(
        tokenize_fn, batched=True,
        remove_columns=remove_cols,
        desc=f"  Tokenizing {task_name}",
    )
    tokenized.set_format("torch")

    train_ds = tokenized["train"]
    eval_ds  = tokenized[cfg["split_eval"]]
    return train_ds, eval_ds


# ============================================================
# 评估指标
# ============================================================

def compute_metric(task_name, preds, labels):
    """统一计算各任务指标，返回 (metric_name, value) 字典"""
    metric_type = GLUE_TASKS[task_name]["metric"]

    if metric_type == "accuracy":
        acc = (np.array(preds) == np.array(labels)).mean()
        return {"accuracy": round(float(acc), 4)}

    elif metric_type == "matthews":
        mcc = matthews_corrcoef(labels, preds)
        return {"matthews_corrcoef": round(float(mcc), 4)}

    elif metric_type == "f1_accuracy":
        f1  = f1_score(labels, preds, average="binary")
        acc = (np.array(preds) == np.array(labels)).mean()
        return {"f1": round(float(f1), 4), "accuracy": round(float(acc), 4)}

    elif metric_type == "pearson_spearman":
        # STS-B：preds 是连续值，不做 argmax
        pr, _ = pearsonr(preds, labels)
        sr, _ = spearmanr(preds, labels)
        return {"pearson": round(float(pr), 4), "spearmanr": round(float(sr), 4)}

    return {}


# ============================================================
# 单任务 fine-tune + 评估
# ============================================================

def run_single_task(task_name, model_path, device, seed=42):
    print(f"\n{'='*55}")
    print(f"任务：{task_name.upper()}")
    print(f"{'='*55}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg      = GLUE_TASKS[task_name]
    ft_cfg   = FINETUNE_CONFIG.get(task_name, FINETUNE_CONFIG["default"])
    is_regression = (cfg["num_labels"] == 1)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 数据
    train_ds, eval_ds = preprocess_glue(task_name, tokenizer)
    if train_ds is None:
        return None

    train_loader = DataLoader(
        train_ds,
        batch_size=ft_cfg["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=ft_cfg["batch_size"] * 2,
        shuffle=False,
    )

    # 模型：从蒸馏后学生权重加载，替换分类头
    bert_config = BertConfig.from_pretrained(model_path)
    bert_config.num_hidden_layers = STUDENT_NUM_LAYERS
    bert_config.num_labels        = cfg["num_labels"]
    # 回归任务用 MSE，分类用 CE（BertForSequenceClassification 内部自动处理）

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        config=bert_config,
        ignore_mismatched_sizes=True,   # 分类头维度不同时忽略
    ).to(device)

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    total_steps = len(train_loader) * ft_cfg["epochs"]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=ft_cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * ft_cfg["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    # ── Fine-tune ─────────────────────────────────────────────
    best_score  = -float("inf")
    best_metric = {}

    for epoch in range(ft_cfg["epochs"]):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"  [{task_name}] Epoch {epoch+1}/{ft_cfg['epochs']}")
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # 回归任务 labels 需要 float
            if is_regression:
                labels = labels.float()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Evaluation ──────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits

                if is_regression:
                    preds = logits.squeeze(-1).cpu().numpy()
                else:
                    preds = logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        metrics = compute_metric(task_name, all_preds, all_labels)

        # 取第一个指标作为 best 判断依据
        primary_score = list(metrics.values())[0]
        is_best = primary_score > best_score
        if is_best:
            best_score  = primary_score
            best_metric = metrics

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  Epoch {epoch+1} | loss={avg_loss:.4f} | {metric_str}"
              + (" ✓ best" if is_best else ""))

    print(f"  [{task_name}] 最优验证指标：{best_metric}")
    return best_metric


# ============================================================
# 汇总打印 GLUE 分数
# ============================================================

def print_glue_summary(results: dict):
    """
    计算并打印 GLUE 总分（取各任务主指标均值，与官方榜单一致）
    主指标定义：
      CoLA → MCC，STS-B → Pearson，MRPC/QQP → F1，其余 → Accuracy
    """
    primary_map = {
        "cola":  "matthews_corrcoef",
        "stsb":  "pearson",
        "mrpc":  "f1",
        "qqp":   "f1",
        "sst2":  "accuracy",
        "mnli":  "accuracy",
        "qnli":  "accuracy",
        "rte":   "accuracy",
        "wnli":  "accuracy",
    }

    print("\n" + "=" * 55)
    print("GLUE 评估汇总")
    print("=" * 55)
    print(f"{'任务':<10} {'主指标':<28} {'分数':>8}")
    print("-" * 55)

    scores = []
    for task, metric_dict in results.items():
        if metric_dict is None:
            print(f"{task.upper():<10} {'跳过（数据加载失败）':<28} {'N/A':>8}")
            continue
        primary_key = primary_map.get(task, list(metric_dict.keys())[0])
        score = metric_dict.get(primary_key, list(metric_dict.values())[0])
        scores.append(score)
        detail = "  ".join(f"{k}={v}" for k, v in metric_dict.items())
        print(f"{task.upper():<10} {detail:<36} {score:>6.4f}")

    if scores:
        avg = np.mean(scores)
        print("-" * 55)
        print(f"{'GLUE Score':<10} {'（各任务主指标均值）':<28} {avg:>8.4f}")
    print("=" * 55)


# ============================================================
# 入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GLUE 评估脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        default=PRETRAIN_STUDENT_SAVE_PATH,
        help="蒸馏后学生模型路径（默认读 config.PRETRAIN_STUDENT_SAVE_PATH）",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(GLUE_TASKS.keys()),
        choices=list(GLUE_TASKS.keys()),
        help="要测试的任务列表，默认全部",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="root/autodl-tmp/glue_score",
        help="将结果保存为 JSON 文件（可选）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"模型路径：{args.model_path}")
    print(f"测试任务：{args.tasks}")
    print(f"设备：{device}\n")

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"模型路径不存在：{args.model_path}\n"
            "请先运行 pretrain_distill.py 完成预训练蒸馏。"
        )

    results = {}
    for task in args.tasks:
        results[task] = run_single_task(
            task_name=task,
            model_path=args.model_path,
            device=device,
            seed=args.seed,
        )

    print_glue_summary(results)

    # 可选：保存 JSON
    if args.output_json:
        import json
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存：{args.output_json}")


if __name__ == "__main__":
    main()