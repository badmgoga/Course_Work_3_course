from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from cross_encoder import dataset

# model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2", num_labels=1)
# tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2")

def preprocess(example):
    query = example['query']
    positives = example['positive_passages']
    negatives = example['negative_passages']

    query = example['query']
    positives = example['positive_passages']
    negatives = example['negative_passages']
    return tokenizer(example["query"], example["passage"], truncation=True, padding="max_length", max_length=256)

encoded = dataset.map(preprocess, batched=True)

args = TrainingArguments(
    "ce_msmarco_finetune",
    evaluation_strategy="steps",
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded,
)

trainer.train()
