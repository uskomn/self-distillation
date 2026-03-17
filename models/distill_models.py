import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForMaskedLM,
    BertForQuestionAnswering,
)
from config import (
    TEACHER_NAME,
    TEACHER_SAVE_PATH,
    PRETRAIN_STUDENT_SAVE_PATH,
    STUDENT_NUM_LAYERS,
    TEACHER_LAYERS_FOR_DISTILL,
)


class TeacherModel(nn.Module):
    """
    教师模型：hfl/chinese-roberta-wwm-ext
    使用时应加载已在 CMRC2018 上微调好的 checkpoint
    """

    def __init__(self, load_finetuned: bool = False):
        super().__init__()

        # ✅ 通过 config 正确开启 attentions/hidden_states 输出
        config = AutoConfig.from_pretrained(TEACHER_NAME)
        config.output_attentions = True
        config.output_hidden_states = True

        if load_finetuned:
            # 蒸馏阶段：加载微调好的教师 checkpoint
            print(f"加载微调后的教师模型：{TEACHER_SAVE_PATH}")
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                TEACHER_SAVE_PATH,
                config=config
            )
        else:
            # 教师微调阶段：加载预训练权重
            print(f"加载预训练教师模型：{TEACHER_NAME}")
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                TEACHER_NAME,
                config=config
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "attentions": outputs.attentions,       # 12层，每层 (B, H, L, L)
            "hidden_states": outputs.hidden_states, # 13个，每个 (B, L, D)
        }


class StudentModel(nn.Module):
    """
    学生模型：6层 BertForQuestionAnswering

    init_mode 控制初始化策略：
      "pretrain_distill" : 从第一阶段预训练蒸馏权重初始化（两阶段蒸馏推荐）
      "finetune_teacher" : 从微调教师偶数层初始化（单阶段蒸馏）
      "scratch"          : 随机初始化
    """

    def __init__(self, init_mode: str = "pretrain_distill"):
        super().__init__()
        assert init_mode in ("pretrain_distill", "finetune_teacher", "scratch"), \
            f"init_mode 必须是 pretrain_distill / finetune_teacher / scratch，当前：{init_mode}"

        config = BertConfig.from_pretrained(TEACHER_NAME)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        config.output_attentions = True
        config.output_hidden_states = True

        self.model = BertForQuestionAnswering(config)

        if init_mode == "pretrain_distill":
            self._init_from_pretrain_distill()
        elif init_mode == "finetune_teacher":
            self._init_from_finetune_teacher()
        # scratch：保持随机初始化

    def _init_from_pretrain_distill(self):
        """
        两阶段蒸馏：从第一阶段预训练蒸馏好的 BertForMaskedLM 权重中取 bert 子模块
        QA head 随机初始化（预训练阶段没有 QA head）
        """
        print(f"[StudentModel] 从预训练蒸馏权重初始化：{PRETRAIN_STUDENT_SAVE_PATH}")
        # 预训练蒸馏保存的是 BertForMaskedLM
        pretrained_mlm = BertForMaskedLM.from_pretrained(PRETRAIN_STUDENT_SAVE_PATH)
        self.model.bert.load_state_dict(pretrained_mlm.bert.state_dict())
        del pretrained_mlm
        torch.cuda.empty_cache()
        print("  ✅ bert encoder/embedding 权重加载完成，QA head 随机初始化")

    def _init_from_finetune_teacher(self):
        """
        单阶段蒸馏：从微调教师的偶数层 + QA head 初始化
        """
        print(f"[StudentModel] 从微调教师权重初始化：{TEACHER_SAVE_PATH}")
        teacher = AutoModelForQuestionAnswering.from_pretrained(TEACHER_SAVE_PATH)

        self.model.bert.embeddings.load_state_dict(
            teacher.bert.embeddings.state_dict()
        )
        for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
            self.model.bert.encoder.layer[student_idx].load_state_dict(
                teacher.bert.encoder.layer[teacher_idx].state_dict()
            )
        self.model.qa_outputs.load_state_dict(teacher.qa_outputs.state_dict())
        del teacher
        torch.cuda.empty_cache()
        print("  ✅ encoder + QA head 权重加载完成")

    # 保留旧接口兼容性
    def _init_from_teacher(self):
        self._init_from_finetune_teacher()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "attentions": outputs.attentions,       # 6层，每层 (B, H, L, L)
            "hidden_states": outputs.hidden_states, # 7个，每个 (B, L, D)
        }
