import torch
import torch.nn as nn
from transformers import (
    AutoModelForQuestionAnswering,
    BertConfig,
    BertForQuestionAnswering
)

TEACHER_NAME = "hfl/chinese-roberta-wwm-ext"


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModelForQuestionAnswering.from_pretrained(
            TEACHER_NAME,
            output_attentions=True,
            output_hidden_states=True
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Teacher 不反向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states
        }


class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = BertConfig.from_pretrained(TEACHER_NAME)

        #核心：减少层数
        config.num_hidden_layers = 6
        config.output_attentions = True
        config.output_hidden_states = True

        self.model = BertForQuestionAnswering(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "attentions": outputs.attentions,
            "hidden_states": outputs.hidden_states
        }