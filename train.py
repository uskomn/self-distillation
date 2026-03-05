# 服务器文件位置/tmp/pycharm_projects_852
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_from_disk
from tqdm import tqdm

from config import (
    TEACHER_NAME,
    TEACHER_SAVE_PATH,
    STUDENT_SAVE_PATH,
    MAX_LENGTH,
    DOC_STRIDE,
    TEACHER_EPOCHS,
    TEACHER_BATCH_SIZE,
    TEACHER_LR,
    TEACHER_WARMUP_RATIO,
    TEACHER_WEIGHT_DECAY,
    STUDENT_EPOCHS,
    STUDENT_BATCH_SIZE,
    STUDENT_LR,
    STUDENT_WARMUP_RATIO,
    STUDENT_WEIGHT_DECAY,
    ALPHA, BETA, GAMMA, TEMPERATURE,
)
from models.distill_models import TeacherModel, StudentModel
from models.distill_loss import total_distillation_loss


# ============================================================
# 数据预处理
# ============================================================

def preprocess_function(examples, tokenizer, is_train=True):
    """
    CMRC2018 数据预处理：滑窗切割长文档，标记答案的 token 位置
    """
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
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]

        # 找到 context 的 token 范围（token_type_ids == 1 的部分）
        sequence_ids = tokenized.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not is_train:
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
            continue

        # 答案字符位置
        answer_start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        answer_end_char = answer_start_char + len(answer_text)

        # 答案是否在当前窗口内
        if (offsets[context_start][0] > answer_end_char or
                offsets[context_end][1] < answer_start_char):
            # 答案不在当前窗口，标记为 [CLS]
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # 找 start token
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start_char:
                token_start += 1
            start_position = token_start - 1

            # 找 end token
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end_char:
                token_end -= 1
            end_position = token_end + 1

            tokenized["start_positions"].append(start_position)
            tokenized["end_positions"].append(end_position)

    return tokenized


def get_dataloaders(tokenizer, batch_size,debug=False):
    """加载 CMRC2018 数据集并返回 DataLoader"""
    print("加载 CMRC2018 数据集...")
    dataset = load_from_disk("/root/autodl-tmp/cmrc2018")
    if debug:
        dataset["train"] = dataset["train"].select(range(200))
        dataset["validation"] = dataset["validation"].select(range(50))

    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer, is_train=True),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["validation"].map(
        lambda x: preprocess_function(x, tokenizer, is_train=True),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小：{len(train_dataset)}，验证集大小：{len(val_dataset)}")
    return train_loader, val_loader


# ============================================================
# 工具函数
# ============================================================

def get_optimizer_and_scheduler(model, train_loader, lr, warmup_ratio, weight_decay, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return optimizer, scheduler


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU：{torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device


# ============================================================
# Step 1：教师模型微调
# ============================================================

def train_teacher():
    print("=" * 60)
    print("Step 1：开始微调教师模型")
    print("=" * 60)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    train_loader, val_loader = get_dataloaders(tokenizer, TEACHER_BATCH_SIZE,debug=False)

    teacher = TeacherModel(load_finetuned=False)

    # 教师微调时需要反向传播，临时去掉 no_grad
    model = teacher.model.to(device)

    optimizer, scheduler = get_optimizer_and_scheduler(
        model, train_loader,
        lr=TEACHER_LR,
        warmup_ratio=TEACHER_WARMUP_RATIO,
        weight_decay=TEACHER_WEIGHT_DECAY,
        epochs=TEACHER_EPOCHS
    )

    best_val_loss = float("inf")

    for epoch in range(TEACHER_EPOCHS):
        # ------ 训练 ------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[教师 Epoch {epoch+1}/{TEACHER_EPOCHS}] 训练")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # ------ 验证 ------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[教师 Epoch {epoch+1}] 验证"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(TEACHER_SAVE_PATH, exist_ok=True)
            model.save_pretrained(TEACHER_SAVE_PATH)
            tokenizer.save_pretrained(TEACHER_SAVE_PATH)
            print(f" 保存最优教师模型，Val Loss: {avg_val_loss:.4f}")

    print(f"\n教师模型微调完成，最优验证损失：{best_val_loss:.4f}")
    print(f"模型已保存至：{TEACHER_SAVE_PATH}")


# ============================================================
# Step 2：学生模型蒸馏训练
# ============================================================

def train_student():
    print("=" * 60)
    print("Step 2：开始蒸馏学生模型")
    print("=" * 60)

    # 检查教师 checkpoint 是否存在
    if not os.path.exists(TEACHER_SAVE_PATH):
        raise FileNotFoundError(
            f"未找到教师模型 checkpoint：{TEACHER_SAVE_PATH}\n"
            f"请先运行 python train.py --mode teacher"
        )

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_SAVE_PATH)
    train_loader, val_loader = get_dataloaders(tokenizer, STUDENT_BATCH_SIZE,debug=False)

    # 加载微调好的教师（推理模式，不参与反向传播）
    teacher = TeacherModel(load_finetuned=True).to(device)
    teacher.eval()

    # 从微调教师权重初始化学生
    student_wrapper = StudentModel(init_from_teacher=True)
    student_model = student_wrapper.model.to(device)

    optimizer, scheduler = get_optimizer_and_scheduler(
        student_model, train_loader,
        lr=STUDENT_LR,
        warmup_ratio=STUDENT_WARMUP_RATIO,
        weight_decay=STUDENT_WEIGHT_DECAY,
        epochs=STUDENT_EPOCHS
    )

    best_val_loss = float("inf")

    for epoch in range(STUDENT_EPOCHS):
        # ------ 训练 ------
        student_model.train()
        total_train_loss = 0.0
        hard_losses, logit_losses, attn_losses = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"[学生 Epoch {epoch+1}/{STUDENT_EPOCHS}] 训练")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # 教师前向（no_grad 在 TeacherModel.forward 内部已处理）
            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # 学生前向
            student_raw = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            student_outputs = {
                "start_logits": student_raw.start_logits,
                "end_logits": student_raw.end_logits,
                "attentions": student_raw.attentions,
                "hidden_states": student_raw.hidden_states,
            }

            # 计算蒸馏损失
            loss, l_hard, l_logits, l_attn = total_distillation_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                start_positions=start_positions,
                end_positions=end_positions,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                alpha=ALPHA,
                beta=BETA,
                gamma=GAMMA,
                temperature=TEMPERATURE,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            hard_losses += l_hard.item()
            logit_losses += l_logits.item()
            attn_losses += l_attn.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                hard=f"{l_hard.item():.4f}",
                logit=f"{l_logits.item():.4f}",
                attn=f"{l_attn.item():.4f}",
            )

        n = len(train_loader)
        print(
            f"Epoch {epoch+1} Train | "
            f"Total: {total_train_loss/n:.4f} | "
            f"Hard: {hard_losses/n:.4f} | "
            f"Logits: {logit_losses/n:.4f} | "
            f"Attn: {attn_losses/n:.4f}"
        )

        # ------ 验证（只用 hard label loss 评估实际 QA 性能）------
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[学生 Epoch {epoch+1}] 验证"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                student_raw = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                val_loss += student_raw.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss (hard): {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(STUDENT_SAVE_PATH, exist_ok=True)
            student_model.save_pretrained(STUDENT_SAVE_PATH)
            tokenizer.save_pretrained(STUDENT_SAVE_PATH)
            print(f"  ✅ 保存最优学生模型，Val Loss: {avg_val_loss:.4f}")

    print(f"\n学生模型蒸馏完成，最优验证损失：{best_val_loss:.4f}")
    print(f"模型已保存至：{STUDENT_SAVE_PATH}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["teacher", "student"],
        help="teacher：微调教师模型；student：蒸馏学生模型"
    )
    args = parser.parse_args()

    if args.mode == "teacher":
        train_teacher()
    elif args.mode == "student":
        train_student()
