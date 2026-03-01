import collections
import numpy as np
import re
import string
from collections import Counter


def postprocess_qa_predictions(examples, features, predictions, tokenizer):
    start_logits, end_logits = predictions

    example_id_to_index = {k["id"]: i for i, k in enumerate(examples)}
    print("Example ids count:", len(example_id_to_index))
    print("Feature example id sample:", features[0]["example_id"])
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]
        ].append(i)

    predictions_dict = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        best_score = -1e9
        best_answer = ""

        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-20:]
            end_indexes = np.argsort(end_logit)[-20:]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    score = start_logit[start_index] + end_logit[end_index]
                    if score > best_score:
                        start_char = offsets[start_index][0]
                        end_char = offsets[end_index][1]
                        best_answer = context[start_char:end_char]
                        best_score = score

        predictions_dict[example["id"]] = best_answer

    return predictions_dict

def normalize_answer(s):
    """去标点 + 去空格"""
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[{}]".format(string.punctuation), "", s)
    return s

def compute_f1(pred, gold):
    pred_tokens = list(normalize_answer(pred))
    gold_tokens = list(normalize_answer(gold))

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_metrics(predictions, references):
    total = len(predictions)
    exact_match = 0
    f1_total = 0

    for example in references:
        pred = predictions[example["id"]]
        gold = example["answers"]["text"][0]

        norm_pred = normalize_answer(pred)
        norm_gold = normalize_answer(gold)

        if norm_pred == norm_gold:
            exact_match += 1

        f1_total += compute_f1(pred, gold)

    return {
        "exact_match": 100 * exact_match / total,
        "f1": 100 * f1_total / total
    }