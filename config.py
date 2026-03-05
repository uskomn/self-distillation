# config.py
# 所有超参数和路径统一管理

# ====== 模型 ======
TEACHER_NAME = "/root/autodl-tmp/pretrained/chinese-roberta"
TEACHER_SAVE_PATH = "/root/autodl-tmp/checkpoints/teacher_cmrc2018"
STUDENT_SAVE_PATH = "/root/autodl-tmp/checkpoints/student_distilled"

# ====== 数据 ======
MAX_LENGTH = 512
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 50

# ====== 教师微调 ======
TEACHER_EPOCHS = 3
TEACHER_BATCH_SIZE = 16
TEACHER_LR = 3e-5
TEACHER_WARMUP_RATIO = 0.1
TEACHER_WEIGHT_DECAY = 0.01

# ====== 学生蒸馏 ======
STUDENT_NUM_LAYERS = 6
STUDENT_EPOCHS = 10
STUDENT_BATCH_SIZE = 16
STUDENT_LR = 5e-5
STUDENT_WARMUP_RATIO = 0.1
STUDENT_WEIGHT_DECAY = 0.01

# ====== 蒸馏损失权重 ======
ALPHA = 1.0          # hard label 损失
BETA = 0.5           # logits 蒸馏损失
GAMMA = 0.1          # attention 蒸馏损失
TEMPERATURE = 2.0    # 蒸馏温度

# ====== 层对齐（教师12层->学生6层，统一用偶数层）======
TEACHER_LAYERS_FOR_DISTILL = [0, 2, 4, 6, 8, 10]
