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
    lora_alpha=32,
    lora_dropout=0.2,
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
        id = example['query_id']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        for p in positives:
            train_data.append({"query": q, "text": p['text'], "label": 1.0, "query_id": id})
        for n in negatives:
            train_data.append({"query": q, "text": n['text'], "label": 0.0, "query_id": id})

train_dataset = Dataset.from_list(train_data)


def preprocess(example):
    result = tokenizer(example["query"], example["text"], truncation=True, padding="max_length", max_length=256)
    result["query_ids"] = example["query_id"]
    return result


train_dataset = train_dataset.map(preprocess, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels", "query_ids"])



valid_subset = dataset["train"].select(range(800, 1000))


valid_data = []
for example in tqdm(valid_subset, desc="Preparing training samples"):
    if not example['has_instruction']:
        q = example['query']
        id = example['query_id']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        for p in positives:
            valid_data.append({"query": q, "text": p['text'], "label": 1.0, "query_id": id})
        for n in negatives:
            valid_data.append({"query": q, "text": n['text'], "label": 0.0, "query_id": id})
valid_dataset = Dataset.from_list(valid_data)


valid_dataset = valid_dataset.map(preprocess, batched=True)
valid_dataset = valid_dataset.rename_column("label", "labels")
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels", "query_ids"])

print(train_dataset[0])
print(valid_dataset[0])

from transformers import TrainingArguments, Trainer



def compute_metrics(eval_pred_with_qids):
    (logits, labels), query_ids = eval_pred_with_qids

    # Приводим к numpy
    logits = logits.detach().cpu().numpy() if torch.is_tensor(logits) else logits
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

    # Группировка по query_ids (как в предыдущем примере)
    from collections import defaultdict
    grouped_logits = defaultdict(list)
    grouped_labels = defaultdict(list)

    for qid, logit, label in zip(query_ids, logits, labels):
        grouped_logits[qid].append(logit)
        grouped_labels[qid].append(label)

    # Далее считаем nDCG и p-MRR по сгруппированным данным
    # ...

    # Вернуть метрики
    return {
        "ndcg@10": ...,
        "p-mrr": ...
    }



from transformers import Trainer
import torch
import numpy as np


class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Вытаскиваем query_ids из inputs
        query_ids = inputs.pop("query_ids")

        # Стандартный вызов предсказания
        outputs = model(**inputs)
        loss = outputs["loss"] if "loss" in outputs else None
        logits = outputs["logits"]

        if prediction_loss_only:
            return (loss, None, None), query_ids

        labels = inputs.get("labels")

        return (loss, logits, labels), query_ids


training_args = TrainingArguments(
    output_dir="lora_crossencoder_no_instruction_21",
    per_device_train_batch_size=32,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,  # или любое удобное число шагов
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    fp16=True,
    report_to="tensorboard"
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.evaluate()