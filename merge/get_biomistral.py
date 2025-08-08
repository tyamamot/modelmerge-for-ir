from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "BioMistral/BioMistral-7B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("./local_BioMistral-7B")
tokenizer.save_pretrained("./local_BioMistral-7B")