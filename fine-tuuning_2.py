from sentence_transformers import CrossEncoder
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import InputExample

# 1. Загрузка модели
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# 2. Загрузка и подготовка данных
# dataset = load_dataset("samaya-ai/msmarco-w-instructions", split="train")

dataset = load_dataset("json", data_files="russian_instruct_dataset_final.jsonl")

subset = dataset["train"].select(range(800))

train_samples = []
for example in tqdm(subset, desc="Preparing training samples"):
    if not example['has_instruction']:
        q = example['query']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        passages = []
        for p in positives:
            train_samples.append(InputExample(texts=[q, p['text']], label=float(1)))
        for n in negatives:
            train_samples.append(InputExample(texts=[q, n['text']], label=float(0)))


# Split (делим на признаки и метки):
# X = [(q, p) for q, p, l in train_samples]
# y = [l for q, p, l in train_samples]

from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# samples - список словарей: [{ "query": ..., "documents": [...], "relevant": [...] }]


samples = dataset["train"].select(range(800, 1000))
valid_samples = []
for example in tqdm(samples, desc="Preparing valid samples"):
    if not example['has_instruction']:
        q = example['query']
        positives = example['positive_passages']
        negatives = example['negative_passages']
        # for p in positives:
        #     valid_samples.append({
        #         "query": q,
        #         "documents": docs,
        #         "positive": positive_indices
        #     })
        docs = [p['text'] for p in positives] + [n['text'] for n in negatives]
        # Индексы релевантных документов: все positives идут первыми
        positive_indices = [0]
        valid_samples.append({
            "query": q,
            "documents": docs,
            "positive": positive_indices
        })

for sample in valid_samples:
    sample["query"] = str(sample["query"])
    sample["documents"] = [str(d) for d in sample["documents"] if d is not None]


for sample in valid_samples:
    assert isinstance(sample["query"], str)
    for doc in sample["documents"]:
        assert isinstance(doc, str)

evaluator = CrossEncoderRerankingEvaluator(
    samples=valid_samples,
    name="valid-eval",
    show_progress_bar=True
)


# 3. Fine-tune через встроенный fit
model.fit(
    train_dataloader=DataLoader(train_samples, batch_size=16, shuffle=True),
    # evaluator=evaluator,
    epochs=1,
    warmup_steps=200,
    # learning_rate=5e-5,
    weight_decay=0.01,
    output_path="rus_no_instruction_2",
    save_best_model=True,
    max_grad_norm=1.0,
    use_amp=True
)
model.save("rus_no_instruction_2")