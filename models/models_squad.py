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
    BertForMaskedLM,
    BertForQuestionAnswering,
)
from config_squad import (
    TEACHER_PRETRAIN_NAME,
    TEACHER_NAME,
    STUDENT_BASE,
    TEACHER_SAVE_PATH,
    STUDENT_SAVE_PATH,
    PT_STUDENT_SAVE_PATH,
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

    init_mode:
      "pretrain_distill" : 从 DistilBERT 式预训练蒸馏权重初始化（两阶段推荐）
      "bert_base"        : 从 bert-base-uncased 前6层初始化（单阶段）
    """
    def __init__(self, init_mode: str = "bert_base", pretrain_path: str = None):
        """
        init_mode:
          "pretrain_distill" : 从预训练蒸馏权重初始化
          "bert_base"        : 从 bert-base-uncased 前6层初始化

        pretrain_path: 预训练蒸馏权重路径，为 None 时使用 config 中的 PT_STUDENT_SAVE_PATH
                       用于消融实验中指定不同组的预训练权重
        """
        super().__init__()
        assert init_mode in ("pretrain_distill", "bert_base"),             f"init_mode 须为 pretrain_distill 或 bert_base，当前：{init_mode}"

        config = BertConfig.from_pretrained(STUDENT_BASE)
        config.num_hidden_layers = STUDENT_NUM_LAYERS
        config.output_attentions = True
        config.output_hidden_states = True
        self.model = BertForQuestionAnswering(config)

        if init_mode == "pretrain_distill":
            self._init_from_pretrain_distill(pretrain_path)
        else:
            self._init_from_bert_base()

    def _init_from_bert_base(self):
        """从 bert-base-uncased 前 6 层初始化，QA head 随机"""
        print(f"从 {STUDENT_BASE} 预训练权重初始化学生...")
        from transformers import AutoModel
        base = AutoModel.from_pretrained(STUDENT_BASE)
        self.model.bert.embeddings.load_state_dict(base.embeddings.state_dict())
        for i in range(STUDENT_NUM_LAYERS):
            self.model.bert.encoder.layer[i].load_state_dict(
                base.encoder.layer[i].state_dict()
            )
        del base
        torch.cuda.empty_cache()
        print("学生初始化完成（bert-base 前6层 + 随机 QA head）")

    def _init_from_pretrain_distill(self, pretrain_path: str = None):
        """
        从预训练蒸馏后的 BertForMaskedLM 权重取出 bert 子模块
        QA head 随机初始化（预训练阶段没有 QA head）

        pretrain_path: 显式指定路径，None 时使用 config 中的 PT_STUDENT_SAVE_PATH
        """
        path = pretrain_path if pretrain_path is not None else PT_STUDENT_SAVE_PATH
        print(f"从预训练蒸馏权重初始化学生：{path}")
        pretrained_mlm = BertForMaskedLM.from_pretrained(path)
        self.model.bert.load_state_dict(pretrained_mlm.bert.state_dict())
        del pretrained_mlm
        torch.cuda.empty_cache()
        print("  bert encoder/embedding 加载完成，QA head 随机初始化")

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
            "attentions":    outputs.attentions,
            "hidden_states": outputs.hidden_states,
        }