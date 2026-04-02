# models_squad.py
"""
SQuAD 实验模型定义

教师：bert-large-uncased-whole-word-masking-finetuned-squad（24层，已在SQuAD微调）
      直接从 HuggingFace 加载，无需再微调
学生：6层 bert-base-uncased，从教师均匀间隔层初始化
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForQuestionAnswering,
)
from config_squad import (
    TEACHER_PRETRAIN_NAME,
    TEACHER_NAME,
    STUDENT_BASE,
    TEACHER_SAVE_PATH,
    STUDENT_NUM_LAYERS,
    TEACHER_LAYERS_FOR_DISTILL,
)


class TeacherModel(nn.Module):
    """
    教师：bert-large-uncased-whole-word-masking
    load_finetuned=False：加载预训练权重，用于微调阶段（参数可更新）
    load_finetuned=True ：加载微调后 checkpoint，用于蒸馏阶段（冻结）
    """
    def __init__(self, load_finetuned: bool = False):
        super().__init__()
        if load_finetuned:
            print(f"加载微调后教师：{TEACHER_SAVE_PATH}")
            source = TEACHER_SAVE_PATH
        else:
            print(f"加载预训练教师：{TEACHER_PRETRAIN_NAME}")
            source = TEACHER_PRETRAIN_NAME

        config = AutoConfig.from_pretrained(source)
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = AutoModelForQuestionAnswering.from_pretrained(source, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        return {
            "start_logits":  outputs.start_logits,
            "end_logits":    outputs.end_logits,
            "attentions":    outputs.attentions,    # 24层，每层 (B, H, L, L)
            "hidden_states": outputs.hidden_states, # 25个，每个 (B, L, D)
        }


class StudentModel(nn.Module):
    """
    学生：6层 BertForQuestionAnswering（基于 bert-base-uncased 结构）
    从教师 24 层中均匀取 6 层初始化：[2, 6, 10, 14, 18, 22]

    注意：
    - 教师 bert-large hidden_size=1024，学生 bert-base hidden_size=768
    - 两者维度不同，无法直接复制 encoder 层权重
    - 因此学生 encoder 从 bert-base-uncased 预训练权重初始化，
      只复制 embedding 层（词表相同，维度不同时跳过）
    - QA head 随机初始化
    """
    def __init__(self):
        super().__init__()

        # 基于 bert-base-uncased 构建 6 层学生
        config = BertConfig.from_pretrained(STUDENT_BASE)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        config.output_attentions = True
        config.output_hidden_states = True

        self.model = BertForQuestionAnswering(config)
        self._init_from_pretrained()

    def _init_from_pretrained(self):
        """
        从 bert-base-uncased 预训练权重初始化学生的前 6 层 encoder
        教师是 large（1024维），学生是 base（768维），维度不同不能直接复制教师层
        用 bert-base 预训练权重是最佳替代：保留语言知识，蒸馏时向教师对齐
        """
        print(f"从 {STUDENT_BASE} 预训练权重初始化学生...")
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(STUDENT_BASE)

        # embedding 层（词表和维度与 bert-base 一致）
        self.model.bert.embeddings.load_state_dict(
            base_model.embeddings.state_dict()
        )
        # encoder 前 6 层
        for i in range(STUDENT_NUM_LAYERS):
            self.model.bert.encoder.layer[i].load_state_dict(
                base_model.encoder.layer[i].state_dict()
            )
        del base_model
        torch.cuda.empty_cache()
        print("学生初始化完成（bert-base 前6层 + 随机 QA head）")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )
        return {
            "start_logits":  outputs.start_logits,
            "end_logits":    outputs.end_logits,
            "attentions":    outputs.attentions,    # 6层
            "hidden_states": outputs.hidden_states, # 7个
        }