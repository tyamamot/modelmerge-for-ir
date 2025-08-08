from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "intfloat/e5-mistral-7b-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

token_embedding = model.get_input_embeddings().weight

torch.save(token_embedding, 'e5-mistral_token_embedding_weights.pth')
model.save_pretrained("./local_e5-mistral-7b")
tokenizer.save_pretrained("./local_e5-mistral-7b")