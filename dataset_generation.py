import json
from tqdm import tqdm
import requests
from datasets import load_dataset


def generate_instruction_yandex(query_text):
    # prompt = f"Напиши пояснение к запросу, чтобы при обучении llm модель с данным пояснением давала более точный ответ пользователю. Сам запрос: \"{query_text}\". Напиши понятную инструкцию."
    prompt = {
        "modelUri": "gpt://b1gk8gfta06c7qsvkpqp/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.65,
            "maxTokens": "500"
        },
        "messages": [
            {
                "role": "system",
                "text": "Ты генерируешь инструкции к промптам для улучшения корректности ответов пользователя."
            },
            {
                "role": "user",
                "text": f"Привет! Сгенерируй четкую инструкцию к данному запросу в 1-2 развернутых предложениях:\n {query_text} \n"
            },
        ]
    }
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key AQVNwbSR7ykqkIoVUaCqc7E9nQUbKtV_Rnpc7Xg5"
    }
    response = requests.post(url, headers=headers, json=prompt)
    result = response.json()
    # print(result)
    return result['result']['alternatives'][0]['message']['text']


def translate_yandex(query_text):
    # prompt = f"Напиши пояснение к запросу, чтобы при обучении llm модель с данным пояснением давала более точный ответ пользователю. Сам запрос: \"{query_text}\". Напиши понятную инструкцию."
    prompt = {
        "modelUri": "gpt://b1gk8gfta06c7qsvkpqp/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.65,
            "maxTokens": "500"
        },
        "messages": [
            {
                "role": "system",
                "text": "Ты переводишь текст с английского на русский."
            },
            {
                "role": "user",
                "text": f"Привет! Переведи мне этот текст:\n {query_text} \n"
            },
        ]
    }
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Api-Key AQVNwbSR7ykqkIoVUaCqc7E9nQUbKtV_Rnpc7Xg5"
    }
    response = requests.post(url, headers=headers, json=prompt)
    result = response.json()
    # print(result)
    return result['result']['alternatives'][0]['message']['text']


base_data = load_dataset("samaya-ai/msmarco-w-instructions", split="train").select(range(1000))

# Пример: исходные данные
# base_data = [
#     # Каждая запись: query, positive_passages, negative_passages
#     {
#         "query": "Что такое сила Архимеда?",
#         "positive_passages": [
#             "Сила Архимеда — это сила, с которой жидкость действует на погруженное в неё тело, выталкивая его вверх."
#         ],
#         "negative_passages": [
#             "Архимед был известным греческим математиком.",
#             "Сила тяжести притягивает все тела к Земле."
#         ]
#     },
#     # ... ещё примеры
# ]


# (Замените на ваш генератор инструкций: здесь для примера список)
def generate_instruction(query):
    # Пример генерации с помощью LLM: обращение к API/модели
    instruction = generate_instruction_yandex(query)
    # Демка - просто шаблоны, для реальных данных замените на реальную генерацию!
    return instruction


d = dict()


def translate(query):
    global d
    if query in d.keys():
        res = d[query]
    else:
        # Пример генерации с помощью LLM: обращение к API/модели
        res = translate_yandex(query)
        d[query] = res
    return res


output = []
for idx, ex in enumerate(tqdm(base_data)):
    query = translate(ex['only_query'])

    if ex['has_instruction']:
        instruction = translate(ex['only_instruction'])
    else:
        instruction = ''

    # instruction = generate_instruction(query)
    item = {
        "query_id": f"ru_{idx+1:}",

        "query": query + ' ' + instruction,
        "only_query": query,
        "instruction": instruction,
        'has_instruction': ex['has_instruction'],
        "positive_passages": [{"docid": f"p{idx+1}", "text": translate(p['text'])} for p in ex["positive_passages"]],
        "negative_passages": [{"docid": f"n{idx+1}_{i}", "text": translate(n['text'])} for i, n in enumerate(ex["negative_passages"])],
        # "instruction_negatives": []  # Для простоты можно добавить генерацию дополнительных нерелевантных с LLM
    }
    output.append(item)

# Сохраняем в jsonl
with open("russian_instruct_dataset_final.jsonl", "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Сгенерировано примеров:", len(output))
