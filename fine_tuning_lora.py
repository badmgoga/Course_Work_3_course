import epoch as epoch
from sentence_transformers import CrossEncoder
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import InputExample
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 1. Загрузка модели
model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Загрузка и подготовка данных
# dataset = load_dataset("samaya-ai/msmarco-w-instructions", split="train")

from peft import LoraConfig, get_peft_model, TaskType


lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=16,
    lora_dropout=0.3,
    bias="none"
)
model = get_peft_model(model, lora_config)

# train_samples = []
# for example in tqdm(subset, desc="Preparing training samples"):
#     if not example['has_instruction']:
#         q = example['query']
#         positives = example['positive_passages']
#         negatives = example['negative_passages']
#         for p in positives:
#             train_samples.append(tokenizer(example["query"], p['text'], truncation=True, padding="max_length", max_length=256))
#         for n in negatives:
#             train_samples.append(tokenizer(example["query"], n['text'], truncation=True, padding="max_length", max_length=256))

from datasets import Dataset

dataset = load_dataset("json", data_files="russian_instruct_dataset_final.jsonl")


train_subset = dataset["train"].select(range(800))

train_data = []

for example in tqdm(train_subset, desc="Preparing training samples"):
    if example['has_instruction']:
        q = example['query']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        for p in positives:
            train_data.append({"query": q, "text": p['text'], "label": 1.0})
        for n in negatives:
            train_data.append({"query": q, "text": n['text'], "label": 0.0})

train_dataset = Dataset.from_list(train_data)


def preprocess(example):
    return tokenizer(example["query"], example["text"], truncation=True, padding="max_length", max_length=256)


train_dataset = train_dataset.map(preprocess, batched=False)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])


valid_subset = dataset["train"].select(range(800, 1000))

valid_data = []

for example in tqdm(valid_subset, desc="Preparing training samples"):
    if example['has_instruction']:
        q = example['query']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        for p in positives:
            valid_data.append({"query": q, "text": p['text'], "label": 1.0})
        for n in negatives:
            valid_data.append({"query": q, "text": n['text'], "label": 0.0})

valid_dataset = Dataset.from_list(valid_data)


valid_dataset = valid_dataset.map(preprocess, batched=False)
valid_dataset = valid_dataset.rename_column("label", "labels")
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])


from transformers import TrainingArguments, Trainer

import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.metrics import ndcg_score

# from collections import defaultdict
#
#
# def group_by_query(queries, logits, labels):
#     grouped_logits = defaultdict(list)
#     grouped_labels = defaultdict(list)
#
#     for q, logit, label in zip(queries, logits, labels):
#         grouped_logits[q].append(logit)
#         grouped_labels[q].append(label)
#
#     # Преобразуем в списки списков
#     grouped_logits = list(grouped_logits.values())
#     grouped_labels = list(grouped_labels.values())
#
#     return grouped_logits, grouped_labels


# def compute_metrics_fn(eval_pred):
#     logits, labels = eval_pred
#     grouped_logits, grouped_labels = group_by_query(queries, logits, labels)
#
#     ndcg_scores = []
#     mrr_scores = []
#
#     for query_logits, query_labels in zip(grouped_logits, grouped_labels):
#         ndcg = ndcg_score([query_labels], [query_logits], k=10)
#         ranks = np.where(np.array(query_labels) == 1)[0]
#         mrr = 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0
#         ndcg_scores.append(ndcg)
#         mrr_scores.append(mrr)
#
#     return {
#         "ndcg@10": np.mean(ndcg_scores),
#         "p-mrr": np.mean(mrr_scores)
#     }

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


training_args = TrainingArguments(
    output_dir="lora_crossencoder_has_instruction_r_16 la_16 ld_0.3",
    per_device_train_batch_size=32,
    # eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=500,  # или любое удобное число шагов
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    fp16=True,
    report_to="tensorboard"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
    # eval_dataset=valid_dataset,
    # compute_metrics=compute_metrics_fn
)


trainer.train()
