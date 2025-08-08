import sys
import os
import ir_datasets

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch import Tensor
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
    model_name = f"../merge/merge_biomistral_e57b_linear_{args.alpha_lower}_{args.alpha_upper}"
    csv_name = f"./score/nfcorpus_score_merge_linear_{args.alpha_lower}_{args.alpha_upper}.csv"
elif args.method == "source":
    model_name = "intfloat/e5-mistral-7b-instruct"
    csv_name = "./score/nfcorpus_score_source.csv"
elif args.method == "lora":
    if not args.lora_name:
        parser.error("--lora_name is required when method is 'lora'.")
    model_name = "intfloat/e5-mistral-7b-instruct"
    lora_name = args.lora_name
    lora_path = f"../tevatron/{lora_name}"
    csv_name = f"./score/nfcorpus_score_lora_{lora_name}.csv"

dataset = ir_datasets.load("beir/nfcorpus/test")


df_query = pd.DataFrame(dataset.queries_iter())

df_qrel = pd.DataFrame(dataset.qrels_iter())

df_doc = pd.DataFrame(dataset.docs_iter())

df_doc['doc'] = df_doc['title'] + ' ' + df_doc['text']


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
task = 'Given a question, retrieve relevant documents that best answer the question'


query_embeddings = []
for index, row in df_query.iterrows():
    q = get_detailed_instruct(task, row["text"])
    batch_dict = tokenizer(q, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)

    embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embedding = F.normalize(embedding, p=2, dim=1)
    query_embeddings.append(embedding.cpu())

df_query["embedding"] = query_embeddings


doc_embeddings = []
for _, row in tqdm(df_doc.iterrows(), total=len(df_doc), desc="Processing documents"):
    batch_dict = tokenizer(row["doc"], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)

    embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    embedding = F.normalize(embedding, p=2, dim=1)
    doc_embeddings.append(embedding.cpu())  

df_doc["embedding"] = doc_embeddings


# Query and document embeddings as lists
query_embeddings = df_query['embedding'].tolist()
doc_embeddings = df_doc['embedding'].tolist()

# Lists to hold the results
query_ids = []
doc_ids = []
scores = []

# Calculate the similarity score for each query-document pair
for i, q_emb in enumerate(query_embeddings):
    for j, d_emb in enumerate(doc_embeddings):
        s = (q_emb @ d_emb.T) * 100
        score = s.item()

        # Store the results
        query_ids.append(df_query.iloc[i]['query_id'])
        doc_ids.append(df_doc.iloc[j]['doc_id'])        
        scores.append(score)

# Create a new DataFrame with query_id, doc_id, and score
results_df = pd.DataFrame({
    'query_id': query_ids,
    'doc_id': doc_ids,
    'score': scores
})

results_df.to_csv(csv_name, index=False)


