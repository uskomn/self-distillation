from datasets import load_dataset

dataset=load_dataset("squad")
dataset.save_to_disk('C:/Users/uskomn/Downloads/squad')