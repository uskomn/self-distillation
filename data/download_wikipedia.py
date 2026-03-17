from datasets import load_dataset
from config import PRETRAIN_DATASET_NAME,PRETRAIN_DATASET_CONFIG

dataset = load_dataset(
        PRETRAIN_DATASET_NAME,
        PRETRAIN_DATASET_CONFIG
    )
dataset.save_to_disk('C:/Users/uskomn/Downloads/wiki_zh_dataset')
