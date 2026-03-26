# evaluate.py
"""
评估模块：计算 EM、F1、推理时间、模型大小、显存占用
支持对单个模型 checkpoint 进行完整评估
"""

import os
import re
import time
import string
import collections
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_from_disk
from tqdm import tqdm

from config import MAX_LENGTH, DOC_STRIDE, MAX_ANSWER_LENGTH


def normalize_answer(s):
    def remove_punc(text):
        exclude = set(string.punctuation + "，。？！；：""''【】《》、")
        return "".join(ch for ch in text if ch not in exclude)

    def remove_whitespace(text):
        return "".join(text.split())

    return remove_whitespace(remove_punc(s))


def get_tokens(s):
    return list(normalize_answer(s))  # 中文按字符分

# em
def compute_exact(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))

# f1
def compute_f1(pred, gold):
    pred_tokens = get_tokens(pred)
    gold_tokens = get_tokens(gold)

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# 模型大小
def get_model_size_mb(model_path: str) -> float:
    """计算 checkpoint 目录下所有模型权重文件的总大小（MB）"""
    total_bytes = 0
    for fname in os.listdir(model_path):
        if fname.endswith(".bin") or fname.endswith(".safetensors"):
            total_bytes += os.path.getsize(os.path.join(model_path, fname))
    return total_bytes / (1024 ** 2)


def get_param_count(model) -> dict:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 2),
    }



# 显存占用
def get_gpu_memory_mb() -> float:
    """返回当前 GPU 已分配显存（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """返回 GPU 峰值显存（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# 数据预处理（验证集，保留 offset_mapping 用于答案还原）
def preprocess_validation(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    tokenized["sample_map"] = sample_map
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_map[i]
        tokenized["example_id"].append(examples["id"][sample_idx])

    return tokenized


def get_val_dataloader(tokenizer, batch_size=32):
    dataset = load_from_disk("/root/autodl-tmp/cmrc2018")
    val_raw = dataset["validation"]

    val_tokenized = val_raw.map(
        lambda x: preprocess_validation(x, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    # 在 set_format 之前，先把 postprocess 需要的元数据列单独保存
    # set_format("torch") 只暴露 tensor 列，example_id / offset_mapping 会被隐藏
    val_meta = {
        "example_id":     val_tokenized["example_id"],       # list[str]
        "offset_mapping": val_tokenized["offset_mapping"],   # list[list[tuple]]
        "token_type_ids": val_tokenized["token_type_ids"],   # list[list[int]]，set_format 前提取
    }

    #  output_all_columns=True 让 DataLoader 只用 tensor 列，但 dataset 索引仍保留全部列
    val_tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "token_type_ids"],
        output_all_columns=False,
    )
    loader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False)

    # 返回 val_meta 供 postprocess 使用，不再依赖遍历 val_tokenized
    return loader, val_raw, val_tokenized, val_meta


# 答案还原（从 logits -> 文本答案）
def postprocess_predictions(val_raw, val_meta, all_start_logits, all_end_logits):
    """
    将 start/end logits 还原为文本答案
    每个 example 取所有滑窗中得分最高的合法 span

    val_meta: dict，包含 example_id (list[str]) 和 offset_mapping (list[list[tuple]])
              由 get_val_dataloader 在 set_format 之前提取，避免 torch format 隐藏非 tensor 列
    """
    features_per_example = collections.defaultdict(list)

    #  直接用 val_meta["example_id"]，不再遍历 val_tokenized
    for i, eid in enumerate(val_meta["example_id"]):
        features_per_example[eid].append(i)

    predictions = {}

    for example in val_raw:
        example_id = example["id"]
        context = example["context"]
        feature_indices = features_per_example[example_id]

        best_answer = {"text": "", "score": float("-inf")}

        for feat_idx in feature_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]

            # 从 val_meta 取 offset_mapping，不依赖 torch format 下的 val_tokenized 索引
            offsets = val_meta["offset_mapping"][feat_idx]

            # 用 token_type_ids 推断 context 边界
            # 从 val_meta 取，不依赖外部 val_tokenized
            token_type_ids_i = val_meta["token_type_ids"][feat_idx]
            context_start = next(
                (i for i, t in enumerate(token_type_ids_i) if t == 1), None
            )
            context_end = next(
                (i for i, t in reversed(list(enumerate(token_type_ids_i))) if t == 1), None
            )
            if context_start is None or context_end is None:
                continue

            # 遍历所有合法 (start, end) 组合
            for start_idx in range(context_start, context_end + 1):
                for end_idx in range(start_idx, min(start_idx + MAX_ANSWER_LENGTH, context_end + 1)):
                    score = start_logits[start_idx] + end_logits[end_idx]
                    if score > best_answer["score"]:
                        char_start = offsets[start_idx][0]
                        char_end = offsets[end_idx][1]
                        best_answer = {
                            "text": context[char_start:char_end],
                            "score": score,
                        }

        predictions[example_id] = best_answer["text"]

    return predictions


# 核心评估函数
def evaluate_model(
        model_path: str,
        experiment_name: str,
        batch_size: int = 32,
        device=None
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"评估实验：{experiment_name}")
    print(f"模型路径：{model_path}")

    # ------ 加载模型 ------
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    # 参数量
    param_info = get_param_count(model)
    print(f"参数量：{param_info['total_params_M']}M")

    # 模型文件大小
    model_size_mb = get_model_size_mb(model_path)
    print(f"模型文件大小：{model_size_mb:.1f} MB")

    model = model.to(device)

    # GPU 清零
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    baseline_gpu_mb = get_gpu_memory_mb()

    # ------ 数据 ------
    loader, val_raw, val_tokenized, val_meta = get_val_dataloader(tokenizer, batch_size)

    # ------ 推理 ------
    all_start_logits = []
    all_end_logits = []
    inference_times = []

    # Warmup（排除首次推理的初始化开销）
    warmup_batch = next(iter(loader))
    with torch.no_grad():
        model(
            input_ids=warmup_batch["input_ids"][:2].to(device),
            attention_mask=warmup_batch["attention_mask"][:2].to(device),
            token_type_ids=warmup_batch["token_type_ids"][:2].to(device),
        )

    # 正式推理计时
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[{experiment_name}] 推理中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()
            inference_times.append((t_end - t_start) / input_ids.size(0))  # 每样本耗时

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # GPU 峰值显存
    peak_gpu_mb = get_peak_gpu_memory_mb()
    inference_gpu_mb = peak_gpu_mb - baseline_gpu_mb

    # ------ 答案还原 & EM/F1 ------
    predictions = postprocess_predictions(val_raw, val_meta, all_start_logits, all_end_logits)

    em_scores, f1_scores = [], []
    for example in val_raw:
        eid = example["id"]
        pred = predictions.get(eid, "")
        gold_answers = example["answers"]["text"]

        em = max(compute_exact(pred, gold) for gold in gold_answers)
        f1 = max(compute_f1(pred, gold) for gold in gold_answers)
        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100
    avg_latency_ms = np.mean(inference_times) * 1000
    p99_latency_ms = np.percentile(inference_times, 99) * 1000

    results = {
        "experiment": experiment_name,
        "model_path": model_path,
        # 准确率
        "EM": round(avg_em, 2),
        "F1": round(avg_f1, 2),
        # 推理时间
        "avg_latency_ms": round(avg_latency_ms, 3),
        "p99_latency_ms": round(p99_latency_ms, 3),
        # 模型大小
        "model_size_mb": round(model_size_mb, 1),
        "param_count_M": param_info["total_params_M"],
        # 资源占用
        "peak_gpu_mb": round(peak_gpu_mb, 1),
        "inference_gpu_mb": round(inference_gpu_mb, 1),
    }

    print(f"\n {experiment_name} 评估结果：")
    print(f"  EM:             {avg_em:.2f}%")
    print(f"  F1:             {avg_f1:.2f}%")
    print(f"  平均延迟:       {avg_latency_ms:.3f} ms/sample")
    print(f"  P99 延迟:       {p99_latency_ms:.3f} ms/sample")
    print(f"  模型大小:       {model_size_mb:.1f} MB")
    print(f"  参数量:         {param_info['total_params_M']}M")
    print(f"  推理峰值显存:   {inference_gpu_mb:.1f} MB")

    return results