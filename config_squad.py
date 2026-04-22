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

# ====== 预训练阶段蒸馏配置 ======
# 教师：bert-base-uncased（12层，冻结，与中文版 RoBERTa 对应）
# 学生：6层，从教师偶数层 [0,2,4,6,8,10] 初始化
PT_TEACHER_NAME = "/root/autodl-tmp/bert-base-uncased"
PT_DATA_SOURCE = "local_xml"  # "wikimedia" 或 "local_xml"
PT_WIKI_NAME = "wikimedia/wikipedia"
PT_WIKI_CONFIG = "20231101.en"  # 英文 Wikipedia
PT_LOCAL_XML_PATH = "/root/autodl-tmp/data/enwiki-20251220-pages-articles-multistream.xml.bz2"
PT_LOCAL_CACHE_DIR = "/root/autodl-tmp/data/enwiki_parsed"
PT_MAX_SAMPLES = 1000000  # 取 100 万条
PT_MLM_PROB = 0.15

# 层对齐策略：教师12层 -> 学生6层，取偶数层 [0,2,4,6,8,10]
PT_TEACHER_LAYERS = [0, 2, 4, 6, 8, 10]

# 预训练蒸馏超参
PT_EPOCHS = 3
PT_BATCH_SIZE = 16
PT_LR = 1e-4
PT_WARMUP_RATIO = 0.06
PT_WEIGHT_DECAY = 0.01

# 预训练蒸馏五项损失权重
# L = λ_mlm·L_MLM + λ_logits·L_logits + λ_emb·L_emb + λ_hidden·L_hidden + λ_attn·L_attn
PT_LAMBDA_MLM       = 1.0   # 学生自身 MLM 损失
PT_LAMBDA_LOGITS    = 0.5   # MLM logits KL 散度（软标签蒸馏）
PT_LAMBDA_EMBEDDING = 0.1   # Embedding 层对齐（MSE，词表示空间对齐）
PT_LAMBDA_HIDDEN    = 0.1   # 逐层隐层 MSE（cosine 归一化）
PT_LAMBDA_ATTN      = 0.1   # 逐层 Attention MSE（sqrt 归一化）

# 预训练蒸馏学生权重保存路径
PT_STUDENT_SAVE_PATH = "/root/autodl-tmp/checkpoints_squad/student_pretrain_distilled"

# ====== 任务感知预训练蒸馏（Task-Aware）======
# 在预训练阶段混入少量 SQuAD QA 数据，让学生提前建立任务意识
# 对应中文版的 CMRC2018 混入策略
PT_QA_INTERLEAVE_EVERY = 4  # 每 N 个 Wiki batch 插入 1 个 QA batch（QA 占比约 20%）
PT_LAMBDA_QA_HARD = 0.5  # QA hard label CE 损失权重
PT_LAMBDA_QA_KD = 0.3  # QA logits KD 损失权重
PT_USE_QA_KD = True  # 是否使用 QA logits 蒸馏（需要微调教师已就绪）

# ====== GLUE 评估配置 ======
GLUE_TASKS = ["sst2", "mrpc", "mnli", "qnli", "qqp", "stsb", "rte", "cola"]
GLUE_EPOCHS = 3
GLUE_BATCH_SIZE = 32
GLUE_LR = 2e-5
GLUE_MAX_LENGTH = 128
GLUE_SAVE_DIR = "/root/autodl-tmp/checkpoints_squad/glue"
GLUE_RESULTS_FILE = "/root/autodl-tmp/results_squad/glue_results.json"
GLUE_WARMUP_RATIO = 0.1
GLUE_WEIGHT_DECAY = 0.01