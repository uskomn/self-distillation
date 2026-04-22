from datasets import load_dataset

tasks = ["cola","sst2","mrpc","stsb","qqp","mnli","qnli","rte"]

for t in tasks:
    ds = load_dataset("glue", t)
    ds.save_to_disk(f"C:/Users/uskomn/Downloads/glue_disk/{t}")