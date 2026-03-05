import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForQuestionAnswering,
)
from config import (
    TEACHER_NAME,
    TEACHER_SAVE_PATH,
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
    学生模型：6层 BERT
    从微调好的教师 checkpoint 的偶数层初始化，加速收敛
    """

    def __init__(self, init_from_teacher: bool = True):
        super().__init__()

        config = BertConfig.from_pretrained(TEACHER_NAME)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        config.output_attentions = True
        config.output_hidden_states = True

        self.model = BertForQuestionAnswering(config)

        if init_from_teacher:
            self._init_from_teacher()

    def _init_from_teacher(self):
        print(f"从微调教师权重初始化学生模型：{TEACHER_SAVE_PATH}")
        teacher = AutoModelForQuestionAnswering.from_pretrained(TEACHER_SAVE_PATH)

        # 复制 embedding 层
        self.model.bert.embeddings.load_state_dict(
            teacher.bert.embeddings.state_dict()
        )

        # 复制 encoder 层：取教师偶数层 [0,2,4,6,8,10] -> 学生 [0,1,2,3,4,5]
        # ✅ 与蒸馏损失中的层对齐策略保持一致
        for student_idx, teacher_idx in enumerate(TEACHER_LAYERS_FOR_DISTILL):
            self.model.bert.encoder.layer[student_idx].load_state_dict(
                teacher.bert.encoder.layer[teacher_idx].state_dict()
            )

        # 复制 QA head（教师已面向任务微调过）
        self.model.qa_outputs.load_state_dict(
            teacher.qa_outputs.state_dict()
        )

        del teacher
        torch.cuda.empty_cache()
        print("学生模型权重初始化完成！")

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
