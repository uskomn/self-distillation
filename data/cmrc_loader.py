import collections
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 384
DOC_STRIDE = 128


def load_cmrc_dataset():
    return load_dataset("cmrc2018")


def prepare_train_features(examples, tokenizer):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    return tokenized_examples


def prepare_validation_features(examples,tokenizer):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]

        tokenized_examples["example_id"].append(
            examples["id"][sample_index]
        )

        # 只保留 context 的 offset
        offset = tokenized_examples["offset_mapping"][i]

        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(offset)
        ]

    return tokenized_examples

def get_dataloader(batch_size=4, small_sample=False):
    dataset = load_cmrc_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if small_sample:
        dataset["train"] = dataset["train"].select(range(500))
        dataset["validation"] = dataset["validation"].select(range(100))

    train_dataset = dataset["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    val_dataset = dataset["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=False
    )

    train_dataset.set_format("torch")

    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True
    )

    return train_dataset, val_dataset, dataset["validation"], tokenizer