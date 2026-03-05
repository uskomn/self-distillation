from datasets import load_dataset

dataset = load_dataset("cmrc2018")

save_path = "./cmrc2018"
dataset.save_to_disk(save_path)

print("CMRC2018 已下载到:", save_path)