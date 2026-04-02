from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

tokenizer.save_pretrained("C:/Users/uskomn/Downloads/bert-base-uncased")
model.save_pretrained("C:/Users/uskomn/Downloads/bert-base-uncased")