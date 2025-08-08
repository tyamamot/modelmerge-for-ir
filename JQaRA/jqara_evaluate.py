import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np
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
    csv_name = f"./score/jqara_score_merge_linear_{alpha_lower}_{alpha_upper}.csv"
elif method == "lora":
    if not args.lora_name:
        parser.error("--lora_name is required when method is 'lora'.")
    lora_name = args.lora_name
    csv_name = f"./score/jqara_score_lora_{lora_name}.csv"
else:
    csv_name = "./score/jqara_score_source.csv"
    
def calculate_ndcg_per_query(df,k):
    ndcg_scores = []

    for q_id, group in df.groupby('q_id'):
        y_true = group['label'].values.reshape(1, -1)
        
        y_score = group['score'].values.reshape(1, -1)

        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)

df_score = pd.read_csv(csv_name)

average_ndcg10 = round(calculate_ndcg_per_query(df_score,10),4)
print(f"alpha_lower = {alpha_lower}, alpha_upper = {alpha_upper}")
print(f'nDCG@10: {average_ndcg10}')




