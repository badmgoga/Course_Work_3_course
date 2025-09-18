---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:12369
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['Сколько тепла требуется для повышения температуры алюминия? Алюминий — популярный металл, используемый в различных отраслях промышленности благодаря своим уникальным свойствам, таким как высокое соотношение прочности к весу, коррозионная стойкость и теплопроводность.\n\nУдельная теплоёмкость материала — важный параметр для понимания его теплового поведения, поскольку она определяет количество тепловой энергии, необходимое для изменения его температуры.\n\nПри оценке документов, связанных с запросом, соответствующий документ должен содержать чёткий и краткий ответ на вопрос, включая конкретный расчёт или формулу для определения количества тепла, необходимого для повышения температуры алюминия. Документ также должен демонстрировать ясное понимание лежащих в основе тепловых принципов и содержать точные числовые значения или преобразования.\n\nКроме того, соответствующий документ должен быть сосредоточен в первую очередь на тепловых свойствах алюминия, с минимальным обсуждением несвязанных тем, таких как гибка или резка алюминиевых листов, или использование алюминия в конкретных отраслях промышленности. В документе также следует избегать обсуждения несвязанных понятий, таких как британские тепловые единицы или тепловые свойства других материалов.', 'Бетта-рыбок следует содержать при температуре от 23 °С до 25 °С. Если вы хотите их разводить, температура должна быть от 26 °С до 27 °С. Выше 27 °С пузырьковое гнездо разрушается, поэтому рекомендуется не поднимать температуру выше этого значения.'],
    ['Кто был моряком в экспедиции Льюиса и Кларка? Как историк, специализирующийся на раннем периоде истории американского Запада, я исследую часто упускаемых из виду участников экспедиции Льюиса и Кларка. Меня особенно интересует роль животных в этом путешествии, поскольку я считаю, что они сыграли решающую роль в успехе экспедиции.\n\nЯ ищу документы, в которых содержатся конкретные сведения о животных, сопровождавших экспедицию, включая их породу, размер и выполняемые ими задачи. В документе также должна быть отражена важность этих животных для успеха экспедиции.\n\nЯ хочу узнать, как эти животные способствовали выживанию команды, будь то охота, защита или другие способы. Соответствующий документ должен привести конкретные примеры действий животных и того, как они повлияли на исход экспедиции. Также в нём должно быть продемонстрировано чёткое понимание роли животных и их значения в контексте экспедиции.\n\nЯ наткнулся на несколько документов, которые кажутся многообещающими, но мне нужна помощь в отборе тех, которые не соответствуют моим критериям. Я ищу документ, в котором не только упоминается животное, но и содержатся конкретные сведения о его участии в экспедиции. Документ должен не просто упоминать животное вскользь, а подробно описывать его опыт и вклад. Если в документе животное упоминается лишь мельком или не содержится никаких конкретных деталей, он не имеет отношения к моему исследованию.\n\nЯ рассчитываю на вашу помощь в поиске документов, которые помогут мне в исследовании и предоставят ценную информацию об участниках экспедиции Льюиса и Кларка, которых часто упускают из виду.', 'Миссия Льюиса и Кларка, названная Джефферсоном «Корпусом открытий», заключалась в том, чтобы найти Северо-Западный проход — водный путь, который искали более ранние исследователи и который соединил бы Атлантический океан с Тихим океаном.'],
    ['какое значение глюкозы считается хорошим Я пациент с диабетом, пытаюсь контролировать уровень сахара в крови, и мне нужна информация о том, какие показатели глюкозы считаются хорошими в зависимости от конкретных числовых диапазонов в разное время суток. Это позволит мне скорректировать диету и приём лекарств. Соответствующий документ должен содержать точные значения в мг/дл до и после приёма пищи.', 'Если сделать анализ крови на глюкозу через два часа после еды, то нормальным уровнем глюкозы в крови считается уровень ниже 140 мг/дл (7,8 ммоль/л) для людей моложе 50 лет. Для людей в возрасте от 50 до 60 лет — ниже 150 мг/дл (8,9 ммоль/л), а для людей 60 лет и старше — ниже 160 мг/дл (8,9 ммоль/л), сообщает ABC News.'],
    ['Определение искусства дадаизма Дадаизм — это культурное и художественное движение, возникшее в начале XX века. Документ актуален, если в нём дано чёткое и краткое определение дадаизма с особым упоминанием его связи с искусством и литературой.', 'существительное. Направление в искусстве, особенно в живописи, начала XX века, в котором отказались от перспективы с единой точкой обзора и использовали простые геометрические формы, взаимопересекающиеся плоскости, а позднее — коллажи. существительное. Направление в искусстве, особенно в живописи, начала XX века, в котором отказались от перспективы с единой точкой обзора и использовали простые геометрические формы, взаимопересекающиеся плоскости, а позднее — коллажи.'],
    ['Какие две сетевые топологии составляют древовидную структуру сети? Топологии сети используются для организации и структурирования компьютерных сетей. Соответствующий документ должен чётко описывать конкретную топологию сети и её состав.', 'Пример: Сеть клиент-сервер. Клиент — это компьютер, на котором пользователи выполняют задачи и делают запросы. Сервер предоставляет информацию или ресурсы клиенту. б) Локальное администрирование: настройка и обслуживание сети должны выполняться на каждом отдельном компьютере, подключённом к сети.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Сколько тепла требуется для повышения температуры алюминия? Алюминий — популярный металл, используемый в различных отраслях промышленности благодаря своим уникальным свойствам, таким как высокое соотношение прочности к весу, коррозионная стойкость и теплопроводность.\n\nУдельная теплоёмкость материала — важный параметр для понимания его теплового поведения, поскольку она определяет количество тепловой энергии, необходимое для изменения его температуры.\n\nПри оценке документов, связанных с запросом, соответствующий документ должен содержать чёткий и краткий ответ на вопрос, включая конкретный расчёт или формулу для определения количества тепла, необходимого для повышения температуры алюминия. Документ также должен демонстрировать ясное понимание лежащих в основе тепловых принципов и содержать точные числовые значения или преобразования.\n\nКроме того, соответствующий документ должен быть сосредоточен в первую очередь на тепловых свойствах алюминия, с минимальным обсуждением несвязанных тем, таких как гибка или резка алюминиевых листов, или использование алюминия в конкретных отраслях промышленности. В документе также следует избегать обсуждения несвязанных понятий, таких как британские тепловые единицы или тепловые свойства других материалов.',
    [
        'Бетта-рыбок следует содержать при температуре от 23 °С до 25 °С. Если вы хотите их разводить, температура должна быть от 26 °С до 27 °С. Выше 27 °С пузырьковое гнездо разрушается, поэтому рекомендуется не поднимать температуру выше этого значения.',
        'Миссия Льюиса и Кларка, названная Джефферсоном «Корпусом открытий», заключалась в том, чтобы найти Северо-Западный проход — водный путь, который искали более ранние исследователи и который соединил бы Атлантический океан с Тихим океаном.',
        'Если сделать анализ крови на глюкозу через два часа после еды, то нормальным уровнем глюкозы в крови считается уровень ниже 140 мг/дл (7,8 ммоль/л) для людей моложе 50 лет. Для людей в возрасте от 50 до 60 лет — ниже 150 мг/дл (8,9 ммоль/л), а для людей 60 лет и старше — ниже 160 мг/дл (8,9 ммоль/л), сообщает ABC News.',
        'существительное. Направление в искусстве, особенно в живописи, начала XX века, в котором отказались от перспективы с единой точкой обзора и использовали простые геометрические формы, взаимопересекающиеся плоскости, а позднее — коллажи. существительное. Направление в искусстве, особенно в живописи, начала XX века, в котором отказались от перспективы с единой точкой обзора и использовали простые геометрические формы, взаимопересекающиеся плоскости, а позднее — коллажи.',
        'Пример: Сеть клиент-сервер. Клиент — это компьютер, на котором пользователи выполняют задачи и делают запросы. Сервер предоставляет информацию или ресурсы клиенту. б) Локальное администрирование: настройка и обслуживание сети должны выполняться на каждом отдельном компьютере, подключённом к сети.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 12,369 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                         | sentence_1                                                                                       | label                                                          |
  |:--------|:---------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                             | string                                                                                           | float                                                          |
  | details | <ul><li>min: 126 characters</li><li>mean: 691.83 characters</li><li>max: 1915 characters</li></ul> | <ul><li>min: 55 characters</li><li>mean: 358.1 characters</li><li>max: 1688 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                    | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Сколько тепла требуется для повышения температуры алюминия? Алюминий — популярный металл, используемый в различных отраслях промышленности благодаря своим уникальным свойствам, таким как высокое соотношение прочности к весу, коррозионная стойкость и теплопроводность.<br><br>Удельная теплоёмкость материала — важный параметр для понимания его теплового поведения, поскольку она определяет количество тепловой энергии, необходимое для изменения его температуры.<br><br>При оценке документов, связанных с запросом, соответствующий документ должен содержать чёткий и краткий ответ на вопрос, включая конкретный расчёт или формулу для определения количества тепла, необходимого для повышения температуры алюминия. Документ также должен демонстрировать ясное понимание лежащих в основе тепловых принципов и содержать точные числовые значения или преобразования.<br><br>Кроме того, соответствующий документ должен быть сосредоточен в первую очередь на тепловых свойствах алюминия, с минимальным обсуждением несвязанных тем, ...</code> | <code>Бетта-рыбок следует содержать при температуре от 23 °С до 25 °С. Если вы хотите их разводить, температура должна быть от 26 °С до 27 °С. Выше 27 °С пузырьковое гнездо разрушается, поэтому рекомендуется не поднимать температуру выше этого значения.</code>                                                                          | <code>0.0</code> |
  | <code>Кто был моряком в экспедиции Льюиса и Кларка? Как историк, специализирующийся на раннем периоде истории американского Запада, я исследую часто упускаемых из виду участников экспедиции Льюиса и Кларка. Меня особенно интересует роль животных в этом путешествии, поскольку я считаю, что они сыграли решающую роль в успехе экспедиции.<br><br>Я ищу документы, в которых содержатся конкретные сведения о животных, сопровождавших экспедицию, включая их породу, размер и выполняемые ими задачи. В документе также должна быть отражена важность этих животных для успеха экспедиции.<br><br>Я хочу узнать, как эти животные способствовали выживанию команды, будь то охота, защита или другие способы. Соответствующий документ должен привести конкретные примеры действий животных и того, как они повлияли на исход экспедиции. Также в нём должно быть продемонстрировано чёткое понимание роли животных и их значения в контексте экспедиции.<br><br>Я наткнулся на несколько документов, которые кажутся многообещающими, но мне нужна помощ...</code> | <code>Миссия Льюиса и Кларка, названная Джефферсоном «Корпусом открытий», заключалась в том, чтобы найти Северо-Западный проход — водный путь, который искали более ранние исследователи и который соединил бы Атлантический океан с Тихим океаном.</code>                                                                                    | <code>0.0</code> |
  | <code>какое значение глюкозы считается хорошим Я пациент с диабетом, пытаюсь контролировать уровень сахара в крови, и мне нужна информация о том, какие показатели глюкозы считаются хорошими в зависимости от конкретных числовых диапазонов в разное время суток. Это позволит мне скорректировать диету и приём лекарств. Соответствующий документ должен содержать точные значения в мг/дл до и после приёма пищи.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <code>Если сделать анализ крови на глюкозу через два часа после еды, то нормальным уровнем глюкозы в крови считается уровень ниже 140 мг/дл (7,8 ммоль/л) для людей моложе 50 лет. Для людей в возрасте от 50 до 60 лет — ниже 150 мг/дл (8,9 ммоль/л), а для людей 60 лет и старше — ниже 160 мг/дл (8,9 ммоль/л), сообщает ABC News.</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 10

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.6460 | 500  | 0.1831        |
| 1.2920 | 1000 | 0.1446        |
| 1.9380 | 1500 | 0.1407        |
| 2.5840 | 2000 | 0.1426        |
| 3.2300 | 2500 | 0.1431        |
| 3.8760 | 3000 | 0.1339        |
| 4.5220 | 3500 | 0.1299        |
| 5.1680 | 4000 | 0.1309        |
| 5.8140 | 4500 | 0.1371        |
| 6.4599 | 5000 | 0.1254        |
| 7.1059 | 5500 | 0.1135        |
| 7.7519 | 6000 | 0.1152        |
| 8.3979 | 6500 | 0.1195        |
| 9.0439 | 7000 | 0.102         |
| 9.6899 | 7500 | 0.1121        |


### Framework Versions
- Python: 3.10.5
- Sentence Transformers: 5.1.0
- Transformers: 4.56.0
- PyTorch: 2.5.1+cu121
- Accelerate: 1.10.0
- Datasets: 4.0.0
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->