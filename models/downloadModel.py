from transformers import AutoTokenizer, AutoModel

model_name = "hfl/chinese-roberta-wwm-ext"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained("./chinese-roberta")
model.save_pretrained("./chinese-roberta")