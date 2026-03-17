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

# ====== 预训练阶段蒸馏 ======
PRETRAIN_STUDENT_SAVE_PATH = "/root/autodl-tmp/checkpoints/student_pretrain_distilled"  # 预训练蒸馏后的学生权重

# 预训练数据源配置
# 可选值：
#   "local_xml"  : 本地手动下载的 Wikipedia XML bz2 文件（当前使用）
#   "wikimedia"  : wikimedia/wikipedia 在线版
#   "cc100"      : cc100 中文子集
PRETRAIN_DATA_SOURCE = "local_xml"

# ====== 本地 XML 数据集路径（修改为你的实际路径）======
PRETRAIN_LOCAL_XML_PATH = "/root/autodl-tmp/data/zhwiki-20251220-pages-articles-multistream.xml.bz2"
# 解析后的纯文本缓存目录（首次运行会生成，之后直接复用）
PRETRAIN_LOCAL_CACHE_DIR = "/root/autodl-tmp/data/zhwiki_parsed"

# wikimedia/wikipedia 在线备选
PRETRAIN_WIKI_NAME   = "wikimedia/wikipedia"
PRETRAIN_WIKI_CONFIG = "20231101.zh"

# cc100 备选
PRETRAIN_CC100_NAME   = "cc100"
PRETRAIN_CC100_CONFIG = "zh-Hans"

PRETRAIN_MAX_SAMPLES = 500000       # 取前 50万条，可按显存/时间调整
PRETRAIN_EPOCHS = 3
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 1e-4
PRETRAIN_WARMUP_RATIO = 0.1
PRETRAIN_WEIGHT_DECAY = 0.01
PRETRAIN_MLM_PROB = 0.15            # MLM 掩码比例

# 预训练阶段蒸馏损失权重
PT_LAMBDA_MLM = 1.0                 # 学生 MLM 任务损失（学生自身学习）
PT_LAMBDA_HIDDEN = 0.1              # 隐层 MSE 蒸馏
PT_LAMBDA_ATTN = 0.1                # Attention MSE 蒸馏
