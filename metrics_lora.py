import torch
from sentence_transformers import CrossEncoder
from datasets import load_dataset
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA доступна! Используется GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна, выполняем на CPU.")


def average_precision(relevance):
    relevant_indices = np.where(relevance == 1)[0]
    if len(relevant_indices) == 0:
        return 0.0
    precisions = []
    for i, idx in enumerate(relevant_indices, start=1):
        precisions.append(i / (idx + 1))
    return np.mean(precisions)


def mean_reciprocal_rank(relevance):
    ranks = np.where(relevance == 1)[0]
    if len(ranks) == 0:
        return 0
    return 1.0 / (ranks[0] + 1)


def predict_scores(model, tokenizer, pairs, device, batch_size=16):
    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            queries = [q for q, _ in batch]
            passages = [p for _, p in batch]
            inputs = tokenizer(queries, passages, padding=True, truncation=True, return_tensors="pt", max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits
            batch_scores = logits.squeeze(-1).detach().cpu().numpy()  # Для бинарной классификации берем логиты класса 1
            scores.extend(batch_scores)
    return scores


def evaluate_metrics(dataset, model, top_k=10):
    all_ap = []
    all_mrr = []
    all_ndcg = []

    for example in tqdm(dataset):
        query = example['query']
        positives = example['positive_passages']
        negatives = example['negative_passages']

        passages = []
        for p in positives:
            passages.append(p['text'])
        for n in negatives:
            passages.append(n['text'])
        labels = [1]*len(positives) + [0]*len(negatives)
        pairs = [(query, p) for p in passages]

        scores = np.array(predict_scores(model, tokenizer, pairs, device=device, batch_size=16))

        sorted_indices = np.argsort(scores)[::-1]
        sorted_labels = np.array(labels)[sorted_indices]
        sorted_scores = scores[sorted_indices]

        ap = average_precision(sorted_labels)
        mrr = mean_reciprocal_rank(sorted_labels)
        ndcg = ndcg_score([sorted_labels], [sorted_scores], k=top_k)

        all_ap.append(ap)
        all_mrr.append(mrr)
        all_ndcg.append(ndcg)

    mean_ap = np.mean(all_ap)
    mean_mrr = np.mean(all_mrr)
    mean_ndcg = np.mean(all_ndcg)
    print(f"MAP: {mean_ap:.4f}")
    print(f"MRR: {mean_mrr:.4f}")
    print(f"nDCG@{top_k}: {mean_ndcg:.4f}")

    return mean_ap, mean_mrr, mean_ndcg


mrr_scores = []
nDCG_scores = []

for i in range(387, 1162, 387):

    # Загрузка датасета (примем для оценки split="test")
    dataset = load_dataset("samaya-ai/msmarco-w-instructions", split="train")

    subset = dataset.select(range(800, 1000))

    # Инициализация модели cross-encoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    base_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    lora_model_path = "lora_crossencoder_has_instruction_r_8 la_16 ld_0.1/checkpoint-" + str(i)

    # Загружаем базовую модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)

    # Загружаем LoRA адаптеры поверх базовой модели
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    print('Эпоха:' + str(i / 387))
    # Запуск оценки
    mean_ap, mean_mrr, mean_ndcg = evaluate_metrics(subset, model, top_k=10)
    mrr_scores.append(mean_mrr)
    nDCG_scores.append(mean_ndcg)



import winsound


# Или проиграть звуковой сигнал частотой 1000 Гц длительностью 500 мс
winsound.Beep(1000, 3000)


import matplotlib.pyplot as plt

x = list(range(1, 11))

plt.plot(x, mrr_scores, marker='o')
plt.xlabel("Кол-во эпох")
plt.ylabel("Значение метрики")
plt.title("Изменение p-MRR")
plt.grid(True)
plt.show()


plt.plot(x, nDCG_scores, marker='o')
plt.xlabel("Кол-во эпох")
plt.ylabel("Значение метрики")
plt.title("Изменение nDCG")
plt.grid(True)
plt.show()
