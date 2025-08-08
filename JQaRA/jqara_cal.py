from datasets import load_dataset
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import argparse
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser(description="Set values for model and csv naming.")
parser.add_argument("--alpha_lower", type=str, help="Alpha value for lower range.")
parser.add_argument("--alpha_upper", type=str, help="Alpha value for upper range.")
parser.add_argument("--method", type=str, choices=["merge", "lora", "source"], required=True,
                    help="Specify method: 'merge', 'lora' or 'source'")
parser.add_argument("--lora_name", type=str, help="Directory name of the LoRA-adapted model.")

args = parser.parse_args()


if args.method == "merge":
    if not args.alpha_lower or not args.alpha_upper:
        parser.error("--alpha_lower and --alpha_upper are required when method is 'merge'.")
    model_name = f"../merge/merge_japanese_e57b_linear_{args.alpha_lower}_{args.alpha_upper}"
    csv_name = f"./score/jqara_score_merge_linear_{args.alpha_lower}_{args.alpha_upper}.csv"
elif args.method == "source":
    model_name = "intfloat/e5-mistral-7b-instruct"
    csv_name = "./score/jqara_score_source.csv"
elif args.method == "lora":
    if not args.lora_name:
        parser.error("--lora_name is required when method is 'lora'.")
    model_name = "intfloat/e5-mistral-7b-instruct"
    lora_name = args.lora_name
    lora_path = f"../tevatron/{lora_name}"
    csv_name = f"./score/jqara_score_lora_{lora_name}.csv"


DATASET_NAME = "hotchpotch/JQaRA"
DATASET_SPLIT = "test"

ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

df = ds.to_pandas()  # type: ignore

df_qrel = df[["q_id", "passage_row_id", "label"]]

df_query = df[["q_id", "question"]].drop_duplicates()

df_doc = df[["passage_row_id", "text"]].drop_duplicates()

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.method == "lora":
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = get_model(lora_path).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

# Load the saved embedding layer weights.
if args.method == "merge":
    saved_embedding_weights = torch.load('../merge/e5-mistral_token_embedding_weights.pth')
    saved_embedding_weights = saved_embedding_weights.to(device)
    current_embedding = model.get_input_embeddings()
    current_embedding.weight.data = saved_embedding_weights
    model.set_input_embeddings(current_embedding)

max_length = 4096

# Each query needs a one-sentence instruction explaining the task
task = 'Given a question, retrieve Wikipedia passages that answer the question'


batch_size = 8 
query_embeddings = []
num_batches = len(df_query) // batch_size + (1 if len(df_query) % batch_size != 0 else 0)

for i in range(num_batches):
    batch_queries = df_query.iloc[i * batch_size:(i + 1) * batch_size]
    q_list = [get_detailed_instruct(task, row["question"]) for _, row in batch_queries.iterrows()]
    
    batch_dict = tokenizer(q_list, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    query_embeddings.extend(embeddings.cpu())

df_query["embedding"] = query_embeddings


batch_size = 4
doc_embeddings = []

num_batches = len(df_doc) // batch_size + (1 if len(df_doc) % batch_size != 0 else 0)

for i in tqdm(range(num_batches), total=num_batches, desc="Processing documents"):
    batch_docs = df_doc.iloc[i * batch_size:(i + 1) * batch_size]["text"].tolist()
    
    batch_dict = tokenizer(batch_docs, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    doc_embeddings.extend(embeddings.cpu())  

df_doc["embedding"] = doc_embeddings


for col in df_query.columns:
    if isinstance(df_query[col].iloc[0], torch.Tensor):
        df_query[col] = df_query[col].apply(lambda x: x.numpy())
        
for col in df_doc.columns:
    if isinstance(df_doc[col].iloc[0], torch.Tensor):
        df_doc[col] = df_doc[col].apply(lambda x: x.numpy())
        
        
scores = []


for _, row in tqdm(df_qrel.iterrows(), total=len(df_qrel)):
    q_id = row["q_id"]
    doc_id = row["passage_row_id"]
    q_emb = df_query[df_query['q_id'] == q_id]['embedding'].values[0]
    d_emb = df_doc[df_doc['passage_row_id'] == doc_id]['embedding'].values[0]
    s = (q_emb @ d_emb.T) * 100
    score = s.item()
    scores.append(score)
df_qrel["score"] = scores


df_qrel.to_csv(csv_name, index=False)




