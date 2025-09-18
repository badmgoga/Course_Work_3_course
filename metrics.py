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

# Загрузка датасета (примем для оценки split="test")
dataset = load_dataset("samaya-ai/msmarco-w-instructions", split="train")

subset = dataset.select(range(1000))

# Инициализация модели cross-encoder
model = CrossEncoder('lora_crossencoder_1', device=device)


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

        scores = model.predict(pairs, batch_size=16)

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


# Запуск оценки
evaluate_metrics(subset, model, top_k=10)
