from beir import util, LoggingHandler

import logging
import pathlib
import sys
import os
import ir_datasets
import pandas as pd
import re
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Set values for csv naming.")
parser.add_argument("--method", type=str, choices=["merge", "lora", "source"], required=True,
                    help="Specify method: 'merge', 'lora' or 'source'")
parser.add_argument("--alpha_lower", type=str, help="Alpha value for lower range.")
parser.add_argument("--alpha_upper", type=str, help="Alpha value for upper range.")
parser.add_argument("--lora_name", type=str, help="Directory name of the LoRA-adapted model.")
args = parser.parse_args()

method = args.method


if method == "merge":
    if not args.alpha_lower or not args.alpha_upper:
        parser.error("--alpha_lower and --alpha_upper are required when method is 'merge'.")
    alpha_lower = args.alpha_lower
    alpha_upper = args.alpha_upper
    csv_name = f"./score/nfcorpus_score_merge_linear_{alpha_lower}_{alpha_upper}.csv"
elif method == "lora":
    if not args.lora_name:
        parser.error("--lora_name is required when method is 'lora'.")
    lora_name = args.lora_name
    csv_name = f"./score/nfcorpus_score_lora_{lora_name}.csv"
else:
    csv_name = "./score/nfcorpus_score_source.csv"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = ir_datasets.load('beir/nfcorpus/test')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), '../testcollections')

df_query = pd.DataFrame(dataset.queries_iter())
print("queries=", len(df_query))
df_qrel = pd.DataFrame(dataset.qrels_iter())
print("qrels=",len(df_qrel))
df_doc = pd.DataFrame(dataset.docs_iter())
print("docs=",len(df_doc))

df_doc['doc'] = df_doc['title'] + ' '+ df_doc['text']


with open(OUTPUT_DIR + "/nfcorpus_queries.jsonl", "w") as f:
    for query in dataset.queries_iter():
        f.write(f'{{"_id": "{query.query_id}", "text": "{query.text}"}}\n')

with open(OUTPUT_DIR + "/nfcorpus_corpus.jsonl", "w") as f:
    for _, row in df_doc.iterrows():
        s = row["doc"]
        s = re.sub(r'\\', r'\\\\', s)
        s = re.sub('"', '\\"', s)
        f.write(f'{{"_id": "{row["doc_id"]}",  "title": "", "text": "{s}"}}\n')

with open(OUTPUT_DIR + "/nfcorpus_qrels.tsv", "w") as f:
    for _,row in df_qrel.iterrows():
        f.write(f'{row["query_id"]}\t{row["doc_id"]}\t{row["relevance"]}\n')


# Load the dataset
corpus_path = OUTPUT_DIR + "/nfcorpus_corpus.jsonl"
query_path = OUTPUT_DIR + "/nfcorpus_queries.jsonl"
qrels_path = OUTPUT_DIR + "/nfcorpus_qrels.tsv"

# Load using load_custom function in GenericDataLoader
corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path,
    query_file=query_path,
    qrels_file=qrels_path).load_custom()


# evauluate
df_score = pd.read_csv(csv_name)
sorted_df = df_score.sort_values(by=['query_id', 'score'], ascending=[True, False])

results1 = {}

for index, row in tqdm(sorted_df.iterrows(), total=len(sorted_df), desc="Processing rows"):
    query_id = str(row["query_id"])
    doc_id = row["doc_id"]
    score = row["score"]
    if query_id not in results1:
        results1[query_id] = {}
    results1[query_id][doc_id] = score
    
#### Evaluate your retrieval using NDCG@k, MAP@K ...
from beir.retrieval.evaluation import EvaluateRetrieval
k_values = [1, 3, 5, 10, 100, 1000]

logging.info("Retriever evaluation for k in: {}".format(k_values))
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results1, k_values)





