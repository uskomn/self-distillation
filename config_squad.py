# config_squad.py
# SQuAD 实验专用配置（D组：Logits + QA-Focused Attention 蒸馏）

# ====== 模型 ======
# 教师：bert-large-uncased-whole-word-masking，在 SQuAD 上微调后作为蒸馏教师
# 学生：bert-base-uncased 6层
TEACHER_PRETRAIN_NAME = "/root/autodl-tmp/bert-large-uncased"  # 教师预训练权重
TEACHER_NAME  = TEACHER_PRETRAIN_NAME   # 兼容旧引用
STUDENT_BASE  = "/root/autodl-tmp/bert-base-uncased"
TEACHER_SAVE_PATH  = "/root/autodl-tmp/checkpoints_squad/teacher_finetuned"  # 微调后保存路径
STUDENT_SAVE_PATH  = "/root/autodl-tmp/checkpoints_squad/student_distilled"

# ====== 教师微调超参 ======
TEACHER_EPOCHS        = 3
TEACHER_BATCH_SIZE    = 8      # large 模型显存占用高，batch 适当减小
TEACHER_LR            = 3e-5
TEACHER_WARMUP_RATIO  = 0.1
TEACHER_WEIGHT_DECAY  = 0.01

# ====== 数据 ======
DATASET_NAME    = "/root/autodl-tmp/squad"      # HuggingFace datasets：rajpurkar/squad（SQuAD 1.1）
MAX_LENGTH      = 256          # SQuAD 官方推荐 384
DOC_STRIDE      = 128
MAX_ANSWER_LENGTH = 30

# ====== 学生结构 ======
STUDENT_NUM_LAYERS = 6
# 教师 bert-large 有 24 层，学生 6 层，取教师偶数层对齐
# 24层 -> 6层：取 [2, 6, 10, 14, 18, 22]（均匀间隔）
TEACHER_LAYERS_FOR_DISTILL = [2, 6, 10, 14, 18, 22]

# ====== 训练超参 ======
EPOCHS          = 5
BATCH_SIZE      = 16
LR              = 3e-5
WARMUP_RATIO    = 0.1
WEIGHT_DECAY    = 0.01

# ====== 蒸馏损失权重 ======
ALPHA       = 1.0    # hard label CE loss
BETA        = 0.5    # logits KD loss
GAMMA       = 0.1    # attention 蒸馏 loss
TEMPERATURE = 2.0