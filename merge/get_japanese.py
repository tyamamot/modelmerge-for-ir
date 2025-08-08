from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "stabilityai/japanese-stablelm-base-gamma-7b"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("./local_japanese-stablelm-base-gamma-7b")
tokenizer.save_pretrained("./local_japanese-stablelm-base-gamma-7b")