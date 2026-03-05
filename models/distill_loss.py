import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TEMPERATURE, ALPHA, BETA, GAMMA, TEACHER_LAYERS_FOR_DISTILL


def hard_label_loss(student_start, student_end, start_positions, end_positions):
    """
    标准交叉熵损失，监督学生的答案边界预测
    """
    ce = nn.CrossEntropyLoss()
    return ce(student_start, start_positions) + ce(student_end, end_positions)


def logits_distillation_loss(
        student_start, student_end,
        teacher_start, teacher_end,
        temperature=TEMPERATURE
):
    """
    经典 KD：KL( softmax(T_t/T) || log_softmax(T_s/T) )
    乘以 T^2 使梯度幅度与温度无关
    """
    T = temperature

    loss_start = F.kl_div(
        F.log_softmax(student_start / T, dim=-1),
        F.softmax(teacher_start / T, dim=-1),
        reduction="batchmean"
    )
    loss_end = F.kl_div(
        F.log_softmax(student_end / T, dim=-1),
        F.softmax(teacher_end / T, dim=-1),
        reduction="batchmean"
    )

    return (loss_start + loss_end) * (T ** 2)


def qa_focused_attention_loss(
        student_attentions,
        teacher_attentions,
        input_ids,
        token_type_ids,
        start_positions,
        end_positions
):
    """
    QA Focused Attention Distillation：
    对 question 区域和 answer 区域的 attention 施加更高权重
    其余区域也参与蒸馏（权重为1），避免梯度稀疏
    """
    mse = nn.MSELoss()
    total_loss = 0.0

    seq_len = token_type_ids.size(1)
    device = token_type_ids.device

    # ====== 构造 question mask（排除 [CLS]=101 和 [SEP]=102）======
    special_tokens = (input_ids == 101) | (input_ids == 102)
    question_mask = ((token_type_ids == 0) & ~special_tokens).float()  # (B, L)

    # ====== 向量化构造 answer mask ======
    # CMRC2018 所有问题均有答案，但仍防御 start > end 的截断边界情况
    indices = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)
    s = start_positions.unsqueeze(1)  # (B, 1)
    e = end_positions.unsqueeze(1)    # (B, 1)
    valid = (s <= e)                  # (B, 1)
    answer_mask = ((indices >= s) & (indices <= e) & valid).float()  # (B, L)

    # ====== 重要区域：question + answer，其余区域基础权重为 1 ======
    # 重要区域权重=2，普通区域权重=1，让模型更关注关键位置同时保留全局信号
    important_mask = torch.clamp(question_mask + answer_mask, 0, 1)  # (B, L)
    base_weight = torch.ones(token_type_ids.size(0), seq_len, seq_len, device=device)
    focus_weight = important_mask.unsqueeze(2) * important_mask.unsqueeze(1)  # (B, L, L)
    weight_matrix = base_weight + focus_weight  # 重要区域权重=2，其余=1，(B, L, L)

    # ====== 逐层计算 attention 蒸馏损失 ======
    for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
        s_attn = student_attentions[student_idx]   # (B, H, L, L)
        t_attn = teacher_attentions[teacher_idx]   # (B, H, L, L)

        weight = weight_matrix.unsqueeze(1)        # (B, 1, L, L) 广播到 head 维度

        # ✅ sqrt 归一化，缓解 attention 分布偏斜（TinyBERT 做法）
        s_attn_norm = torch.sqrt(s_attn + 1e-6)
        t_attn_norm = torch.sqrt(t_attn.detach() + 1e-6)

        loss = mse(s_attn_norm * weight, t_attn_norm * weight)
        total_loss += loss

    return total_loss / len(TEACHER_LAYERS_FOR_DISTILL)


def total_distillation_loss(
        student_outputs,
        teacher_outputs,
        start_positions,
        end_positions,
        input_ids,
        token_type_ids,
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        temperature=TEMPERATURE,
):
    """
    总蒸馏损失 = alpha * L_hard + beta * L_logits + gamma * L_attn
    """
    l_hard = hard_label_loss(
        student_outputs["start_logits"], student_outputs["end_logits"],
        start_positions, end_positions
    )

    l_logits = logits_distillation_loss(
        student_outputs["start_logits"], student_outputs["end_logits"],
        teacher_outputs["start_logits"], teacher_outputs["end_logits"],
        temperature
    )

    l_attn = qa_focused_attention_loss(
        student_outputs["attentions"], teacher_outputs["attentions"],
        input_ids, token_type_ids,
        start_positions, end_positions
    )

    total = alpha * l_hard + beta * l_logits + gamma * l_attn

    return total, l_hard, l_logits, l_attn
