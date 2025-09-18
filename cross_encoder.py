from collections import defaultdict

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from datasets import load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import numpy as np

dataset = load_dataset("samaya-ai/msmarco-w-instructions", split="train")

subset = dataset.select(range(100000))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("CUDA доступна! Используется GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA недоступна, выполняем на CPU.")

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device=device)

data, labels = [], []
for row in tqdm(subset):
    q = row["query"]
    instr = row.get("only_instruction") or ""
    q_full = f"{instr} {q}".strip()
    for pos in row["positive_passages"]:
        data.append([q_full, pos["text"]])
        labels.append(1)
    data.append([q_full, row["negative_passages"][0]["text"]])
    labels.append(0)
    # for neg in row["negative_passages"]:
    #     data.append([q_full, neg["text"]])
    #     labels.append(0)

scores = model.predict(data, batch_size=32, show_progress_bar=True)
pred_labels = [int(s >= 0) for s in scores]

print("Accuracy:", accuracy_score(labels, pred_labels))
print("F1:", f1_score(labels, pred_labels))
print("ROC-AUC:", roc_auc_score(labels, scores))

y_true = np.array(labels)
y_scores = np.array(scores)
mean_ndcg = ndcg_score([y_true], [y_scores], k=10)
print(f"nDCG@10 (упрощённый по всей выборке): {mean_ndcg:.4f}")

# grouped_labels = defaultdict(list)
# grouped_scores = defaultdict(list)
#
# for row, label, score in zip(dataset, labels, scores):
#     instr = row.get("only_instruction") or ""
#     q_full = f"{instr} {row['query']}".strip()
#     grouped_labels[q_full].append(label)
#     grouped_scores[q_full].append(score)
#
#
# ndcgs = []
# for q in grouped_labels:
#     y_true = [grouped_labels[q]]
#     y_score = [grouped_scores[q]]

#     if len(grouped_labels[q]) > 1:  # Проверяем, что документов больше одного
#         ndcgs.append(ndcg_score(y_true, y_score, k=10))
#     else:
#         # Можно пропускать или добавить логику обработки
#         print(f"Пропускаем запрос {q} с 1 документом")
#
# mean_ndcg = np.mean(ndcgs)
# print(f"mean nDCG@10 по запросам: {mean_ndcg:.4f}")
