from sentence_transformers import CrossEncoder
from datasets import load_dataset
import numpy as np

# Загружаем датасет MS MARCO Average (v2.1, validation часть)
dataset = load_dataset("ms_marco", "v2.1", split="validation[:200]")

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

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
        return 0.0
    return 1.0 / (ranks[0] + 1)

MAP_list = []
MRR_list = []

for example in dataset:
    query = example['query']

    # passages как словарь с ключами passage_text и is_selected
    passages_texts = example['passages']['passage_text']
    is_selected = example['passages']['is_selected']  # список бинарных меток

    # Формируем пары (запрос, пассаж)
    pairs = [(query, passage) for passage in passages_texts]

    # Получаем предсказанные моделью оценки
    scores = model.predict(pairs, batch_size=16)

    # Сортируем по убыванию
    sorted_indices = np.argsort(scores)[::-1]

    # Сортируем метки релевантности в соответствии со скором
    sorted_relevance = np.array(is_selected)[sorted_indices]

    # Считаем метрики
    MAP_list.append(average_precision(sorted_relevance))
    MRR_list.append(mean_reciprocal_rank(sorted_relevance))

mean_map = np.mean(MAP_list)
mean_mrr = np.mean(MRR_list)

print(f"MAP: {mean_map:.4f}")
print(f"p-MRR: {mean_mrr:.4f}")
