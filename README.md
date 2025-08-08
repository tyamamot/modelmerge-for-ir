# Effect of Model Merging in Domain-Specific Ad-hoc Retrieval

This is the repository for the paper:
> Taiga Sasaki, Takehiro Yamamoto, Hiroaki Ohshima and Sumio Fujita:  
> *Effect of Model Merging in Domain-Specific Ad-hoc Retrieval*. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025),   
> November 2025.  

To reproduce the results reported in the paper, please follow the instructions below.


## 1. Preparation

### Download Models
Download the necessary models:
```sh
cd merge
python get_e57b.py
python get_biomistral.py
python get_japanese.py
```

### Unzip Data Files
Unzip all `.zip` files in the `data/` directory:
```sh
cd data
for file in *.zip; do
    unzip "$file"
done
```

### Merge Models
Merge the models using `mergekit`.
The config files for `mergekit` are available at `merge/*.yaml`.
 
The following command creates the merged model which merges
`e5-mistral-7b-instruct` and `BioMistral-7B` with hyperparameters $\alpha_{lower}=0.75$ and $\alpha_{upper}=1.0$
.
```sh
cd merge
git clone https://github.com/arcee-ai/mergekit.git 
cd mergekit
pip install -e .
cd ..
mergekit-yaml ./merge_config_biomistral_e57b_linear_075_10.yaml --cuda ./merge_biomistral_e57b_linear_075_10
```
The hyperparameters (e.g., $\alpha_{lower}=0.75$, $\alpha_{upper}=1.0$ in the above example)
were optimized via grid search for each experimental setting, and are specified in the corresponding YAML files under `configs/merge/`.
### LoRA Fine-tuning 
We also provide instructions for fine-tuning the base model `e5-mistral-7b-instruct` using LoRA (Low-Rank Adaptation).

#### Clone Tevatron and Install
We use Tevatron for fine-tuning. Please clone the [Tevatron repository](https://github.com/texttron/tevatron) under the root directory of this repository (i.e., alongside `merge/`, `data/`, etc.), and follow the installation instructions provided in the official README.

#### Example: LoRA Fine-tuning on NFCorpus
To fine-tune on NFCorpus (full dataset), run:

```sh
deepspeed --include localhost:0 --module tevatron.retriever.driver.train \
    --deepspeed deepspeed/ds_zero3_config.json \
    --output_dir nfcorpus_lora_full_data \
    --model_name_or_path intfloat/e5-mistral-7b-instruct \
    --lora \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps 1000 \
    --dataset_path ../data/NFCorpus/nfcorpus-dev-1000q-all.jsonl \
    --query_prefix "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery:" \
    --fp16 \
    --pooling eos \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --train_group_size 16 \
    --learning_rate 1e-4 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --max_steps 63 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 4
```
#### Note
- `--max_steps` and `--num_train_epochs` were determined via early stopping for each experimental setting. Please refer to the corresponding YAML files under `configs/lora/` to set these values appropriately.
- The `--query_prefix` should be adapted based on the dataset. Refer to the table below:

| Dataset      | query prefix                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------- |
| **NFCorpus** | `Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery:`   |
| **SciFact**  | `Instruct: Given a scientific claim, retrieve documents that support or refute the claim\nQuery:` |
| **MIRACL**   | `Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:`        |
| **JQaRA**    | `Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:`        |

- You may also apply this procedure to other datasets by modifying `--dataset_path`, `--output_dir`.

## 2. Evaluation
All evaluation scripts support the following usage:

- For merged models, specify `--method merge --alpha_lower <value> --alpha_upper <value>`

- For LoRA-finetuned models, specify `--method lora --lora_name <lora_dir_name>`

- For source model `e5-mistral-7b-instruct`, specify only `--method source`

Valid values for `<value>` are `00`, `025`, `05`, `075`, and `10`.

### NFCorpus
#### Merged model:
```sh
cd NFCorpus
python nfcorpus_cal.py --method merge --alpha_lower 075 --alpha_upper 10
python nfcorpus_evaluate.py --method merge --alpha_lower 075 --alpha_upper 10
```
#### LoRA-finetuned model:
```sh
cd NFCorpus
python nfcorpus_cal.py --method lora --lora_name <lora_dir_name>
python nfcorpus_evaluate.py --method lora --lora_name <lora_dir_name>
```
#### Source model:
```sh
cd NFCorpus
python nfcorpus_cal.py --method source
python nfcorpus_evaluate.py --method source
```

### SciFact
#### Merged model:
```sh
cd SciFact
python scifact_cal.py --method merge --alpha_lower 075 --alpha_upper 10
python scifact_evaluate.py --method merge --alpha_lower 075 --alpha_upper 10
```
#### LoRA-finetuned model:
```sh
cd SciFact
python scifact_cal.py --method lora --lora_name <lora_dir_name>
python scifact_evaluate.py --method lora --lora_name <lora_dir_name>
```
#### Source model:
```sh
cd SciFact
python scifact_cal.py --method source
python scifact_evaluate.py --method source
```
### MIRACL
#### Merged model:
```sh
cd MIRACL
python miracl_cal.py --method merge --alpha_lower 075 --alpha_upper 075
python miracl_evaluate.py --method merge --alpha_lower 075 --alpha_upper 075
```
#### LoRA-finetuned model:
```sh
cd MIRACL
python miracl_cal.py --method lora --lora_name <lora_dir_name>
python miracl_evaluate.py --method lora --lora_name <lora_dir_name>
```
#### Source model:
```sh
cd MIRACL
python miracl_cal.py --method source
python miracl_evaluate.py --method source
```
### JQaRA
#### Merged model:
```sh
cd JQaRA
python jqara_cal.py --method merge --alpha_lower 05 --alpha_upper 10
python jqara_evaluate.py --method merge --alpha_lower 05 --alpha_upper 10
```
#### LoRA-finetuned model:
```sh
cd JQaRA
python jqara_cal.py --method lora --lora_name <lora_dir_name>
python jqara_evaluate.py --method lora --lora_name <lora_dir_name>
```
#### Source model:
```sh
cd JQaRA
python jqara_cal.py --method source
python jqara_evaluate.py --method source
```
## Note
- The steps for installing packages, downloading models, and merging models are preparatory tasks. Complete these steps before proceeding with evaluation.

## How to Cite
If you use this code or data, please cite it as follows:
> Taiga Sasaki, Takehiro Yamamoto, Hiroaki Ohshima and Sumio Fujita:  
> *Effect of Model Merging in Domain-Specific Ad-hoc Retrieval*. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025),   
> November 2025.

## Contact
For any inquiries, please contact:

**Takehiro Yamamoto, University of Hyogo**  
Email: [t.yamamoto@sis.u-hyogo.ac.jp](t.yamamoto@sis.u-hyogo.ac.jp)