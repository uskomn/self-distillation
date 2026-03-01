import torch
import torch.nn as nn
import torch.nn.functional as F

# 经典logits蒸馏
def logits_distillation_loss(
        student_start,
        student_end,
        teacher_start,
        teacher_end,
        temperature=2.0
):
    """
    经典 KD:
    KL( softmax(T_teacher/T), softmax(T_student/T) )
    """

    T = temperature

    student_start_logprob = F.log_softmax(student_start / T, dim=-1)
    student_end_logprob = F.log_softmax(student_end / T, dim=-1)

    teacher_start_prob = F.softmax(teacher_start / T, dim=-1)
    teacher_end_prob = F.softmax(teacher_end / T, dim=-1)

    loss_start = F.kl_div(
        student_start_logprob,
        teacher_start_prob,
        reduction="batchmean"
    )

    loss_end = F.kl_div(
        student_end_logprob,
        teacher_end_prob,
        reduction="batchmean"
    )

    return (loss_start + loss_end) * (T ** 2)

def hard_label_loss(
        student_start,
        student_end,
        start_positions,
        end_positions
):
    ce_loss = nn.CrossEntropyLoss()

    loss_start = ce_loss(student_start, start_positions)
    loss_end = ce_loss(student_end, end_positions)

    return loss_start + loss_end

def qa_focused_attention_loss(
        student_attentions,
        teacher_attentions,
        token_type_ids,
        start_positions,
        end_positions
):
    """
    QA Focused Attention Distillation
    """

    mse = nn.MSELoss()
    total_loss = 0.0

    # teacher 12层 -> student 6层
    teacher_layers = [1, 3, 5, 7, 9, 11]

    batch_size = token_type_ids.size(0)
    seq_len = token_type_ids.size(1)

    # 构造 question mask
    question_mask = (token_type_ids == 0).float()

    # 构造 answer mask
    answer_mask = torch.zeros_like(token_type_ids).float()

    for b in range(batch_size):
        answer_mask[b, start_positions[b]: end_positions[b] + 1] = 1.0

    # 合并关键区域
    important_mask = torch.clamp(question_mask + answer_mask, 0, 1)

    # 构造二维权重矩阵
    weight_matrix = important_mask.unsqueeze(2) * important_mask.unsqueeze(1)

    # shape: (batch, seq, seq)

    for i, t_layer in enumerate(teacher_layers):
        s_attn = student_attentions[i]          # (B, H, L, L)
        t_attn = teacher_attentions[t_layer]   # (B, H, L, L)

        # 扩展权重到 head 维度
        weight = weight_matrix.unsqueeze(1)

        loss = mse(s_attn * weight,
                   t_attn.detach() * weight)

        total_loss += loss

    return total_loss / len(teacher_layers)

def total_distillation_loss(
        student_outputs,
        teacher_outputs,
        start_positions,
        token_type_ids,
        end_positions,
        alpha=1.0,
        beta=0.7,
        gamma=0.3,
        temperature=2.0
):
    l_hard = hard_label_loss(
        student_outputs["start_logits"],
        student_outputs["end_logits"],
        start_positions,
        end_positions
    )

    l_logits = logits_distillation_loss(
        student_outputs["start_logits"],
        student_outputs["end_logits"],
        teacher_outputs["start_logits"],
        teacher_outputs["end_logits"],
        temperature
    )

    l_attn = qa_focused_attention_loss(
        student_outputs["attentions"],
        teacher_outputs["attentions"],
        token_type_ids,
        start_positions,
        end_positions
    )

    total = alpha * l_hard + beta * l_logits + gamma * l_attn

    return total, l_hard, l_logits, l_attn