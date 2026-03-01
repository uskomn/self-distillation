import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.metrics import postprocess_qa_predictions, compute_metrics
import numpy as np

from data.cmrc_loader import get_dataloader
from models.distill_models import TeacherModel, StudentModel
from models.distill_loss import (
    qa_focused_attention_loss,
    logits_distillation_loss,
    hard_label_loss
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="full",
                    choices=["student", "kl", "full"])
args = parser.parse_args()

experiment_type = args.exp
print("当前实验模式:", experiment_type)

BATCH_SIZE = 4
EPOCHS = 1
LR = 3e-5
ALPHA = 1.0
BETA = 0.7
GAMMA = 0.3
TEMPERATURE = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, raw_val_dataset, tokenizer = get_dataloader(
    batch_size=BATCH_SIZE,
    small_sample=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

student = StudentModel().to(device)

if experiment_type in ["kl", "full"]:
    teacher = TeacherModel().to(device)
    teacher.eval()
else:
    teacher = None

optimizer = AdamW(student.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


for epoch in range(EPOCHS):

    student.train()
    total_loss_epoch = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        else:
            token_type_ids = torch.zeros_like(input_ids).to(device)


        student_outputs = student(input_ids, attention_mask)

        # Hard label loss（始终有）
        l_hard = hard_label_loss(
            student_outputs["start_logits"],
            student_outputs["end_logits"],
            start_positions,
            end_positions
        )


        if experiment_type == "student":
            loss = l_hard
            l_logits = torch.tensor(0.0)
            l_attn = torch.tensor(0.0)

        elif experiment_type == "kl":

            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask)

            l_logits = logits_distillation_loss(
                student_outputs["start_logits"],
                student_outputs["end_logits"],
                teacher_outputs["start_logits"],
                teacher_outputs["end_logits"],
                TEMPERATURE
            )

            loss = ALPHA * l_hard + BETA * l_logits
            l_attn = torch.tensor(0.0)

        elif experiment_type == "full":

            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask)

            l_logits = logits_distillation_loss(
                student_outputs["start_logits"],
                student_outputs["end_logits"],
                teacher_outputs["start_logits"],
                teacher_outputs["end_logits"],
                TEMPERATURE
            )

            l_attn = qa_focused_attention_loss(
                student_outputs["attentions"],
                teacher_outputs["attentions"],
                token_type_ids,
                start_positions,
                end_positions
            )

            loss = ALPHA * l_hard + BETA * l_logits + GAMMA * l_attn

        # =====================
        # 反向传播
        # =====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_epoch += loss.item()

        progress_bar.set_postfix({
            "loss": loss.item(),
            "hard": l_hard.item(),
            "logits": l_logits.item(),
            "attn": l_attn.item()
        })


    student.eval()

    all_start_logits = []
    all_end_logits = []

    with torch.no_grad():
        for i in tqdm(range(0, len(val_dataset), BATCH_SIZE), desc="Validation"):

            batch = val_dataset[i: i + BATCH_SIZE]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = student(input_ids, attention_mask)

            all_start_logits.append(outputs["start_logits"].cpu().numpy())
            all_end_logits.append(outputs["end_logits"].cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)

    predictions = postprocess_qa_predictions(
        raw_val_dataset,
        val_dataset,
        (all_start_logits, all_end_logits),
        tokenizer
    )

    metrics = compute_metrics(predictions, raw_val_dataset)

    print("Validation EM:", metrics["exact_match"])
    print("Validation F1:", metrics["f1"])


save_path = f"student_{experiment_type}.pt"
torch.save(student.state_dict(), save_path)
print("模型已保存:", save_path)