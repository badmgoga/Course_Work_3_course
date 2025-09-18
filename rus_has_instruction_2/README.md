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
    ['что такое акст Я планирую поездку на Аляску и хочу узнать о часовом поясе, конкретно о стандартном часовом поясе, используемом в Северной Америке, который отстаёт от UTC на 9 часов. Я хочу знать название этого часового пояса и его смещение от UTC.', 'Это было связано с тем, что в тропическом году 365,2422 солнечных дней или 366,2422 сидерических дней, пока я не выяснил, что на самом деле в тропическом году 365,2421822916 солнечных дней или 366,2421822916 сидерических дней. 365 дней 5 часов 48 минут 44 секунды и 55 сотых долей секунды в году или 366 сидерических дней 5 сидерических часов 48 сидерических минут 44 сидерических секунды и 55 сидерических сотых долей секунды в году по сравнению с 365 днями 5 часами 48 минутами 46 секундами и 8...'],
    ['Как взять карту пополнения счёта у MTN Соответствующий документ должен содержать конкретный поэтапный процесс получения карты пополнения баланса у поставщика телекоммуникационных услуг, включая все необходимые коды набора или пункты меню. В документе также должны быть указаны минимальная и максимальная суммы, которые можно получить в долг, а также любые связанные с этим комиссии или сборы. Описанный процесс должен применяться только к абонентам с предоплатой.', 'Итак, откуда берутся грунтовые воды? Очевидно, что крупные системы подземных вод формируются очень медленно. И они находятся в очень хрупком балансе между использованием и пополнением.\n\nОПРЕДЕЛИМ: зона пополнения — место, где вода поступает в водоносный горизонт.\n\nВо многих районах грунтовые воды можно считать невозобновляемым ресурсом. Это особенно актуально для глубоких водоносных горизонтов.'],
    ['что такое люк striker Накладки на люки — это важнейший компонент различных систем, включая транспортные средства и механические устройства. В этих системах накладка на люк играет жизненно важную роль в обеспечении правильного выравнивания и надёжного закрытия. Соответствующий документ должен содержать подробное описание состава накладки на люк, её функций и того, как она взаимодействует с другими компонентами для достижения определённой цели. Также в документе должно быть объяснено, как состояние накладки на люк может повлиять на общую производительность системы. Соответствующий документ должен быть сосредоточен на технических аспектах накладки на люк, приводя конкретные примеры или иллюстрации в поддержку своих утверждений.', 'Металлическая пластина, прикреплённая к дверному косяку, с отверстием или отверстиями для дверного засова. Пластина защелкивается на засове при его активации. Уплотнитель. Энергоэффективная и устойчивая к атмосферным воздействиям система уплотнителей в нижней части двери, которая обеспечивает герметичность и препятствует проникновению воздуха и воды между дверью и порогом.'],
    ['сколько унций в чашке В соответствующем документе должен быть приведён прямой коэффициент пересчёта между чашками и жидными унциями, а также чётко и кратко указана эквивалентность. Документ также должен содержать дополнительный контекст или пояснения, подтверждающие правильность пересчёта. Коэффициент пересчёта должен быть представлен в простом и понятном формате. Документы, которые предлагают только косвенные или неявные преобразования, или те, которые требуют дополнительных расчётов или поиска, не являются релевантными.', 'Сообщить о нарушении. 1  1 чашка = 8 унций, следовательно, 1/4 чашки — это 2 унции. 2  1 чашка = 8 унций, следовательно, 1/4 чашки — это 2 унции. 3  2 унции.  2 унции жидкости. 2  унц.'],
    ['сумма для имеющих право иждивенцев 2016 Значимый документ — это документ, в котором чётко указаны конкретные суммы или диапазон сумм, которые можно внести на счёт для ухода за зависимыми лицами в определённом году. Также в нём должна быть информация о критериях соответствия для таких счетов. Документы, в которых обсуждаются только общие налоговые льготы или вычеты для зависимых лиц без упоминания конкретных сумм взносов или критериев соответствия, не являются значимыми. Кроме того, документы, посвящённые налоговым декларациям или статусу подачи декларации, также не являются значимыми.', 'Это определяется следующим образом. Максимальная сумма расходов на уход за иждивенцами, связанных с работой, которая может быть зачтена при расчёте налогового кредита, составляет 3 000 долларов США. Процентная сумма (от 20 % до 35 %), определяемая вашим доходом, умножается на эту сумму для расчёта налогового кредита. Таким образом, семья с скорректированным валовым доходом в 45 000 долларов США, которая потратила не менее 3 000 долларов США на расходы, связанные с работой по уходу, получит налоговый кредит в размере 600 долларов США (3 000 долларов США х 20 %).'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'что такое акст Я планирую поездку на Аляску и хочу узнать о часовом поясе, конкретно о стандартном часовом поясе, используемом в Северной Америке, который отстаёт от UTC на 9 часов. Я хочу знать название этого часового пояса и его смещение от UTC.',
    [
        'Это было связано с тем, что в тропическом году 365,2422 солнечных дней или 366,2422 сидерических дней, пока я не выяснил, что на самом деле в тропическом году 365,2421822916 солнечных дней или 366,2421822916 сидерических дней. 365 дней 5 часов 48 минут 44 секунды и 55 сотых долей секунды в году или 366 сидерических дней 5 сидерических часов 48 сидерических минут 44 сидерических секунды и 55 сидерических сотых долей секунды в году по сравнению с 365 днями 5 часами 48 минутами 46 секундами и 8...',
        'Итак, откуда берутся грунтовые воды? Очевидно, что крупные системы подземных вод формируются очень медленно. И они находятся в очень хрупком балансе между использованием и пополнением.\n\nОПРЕДЕЛИМ: зона пополнения — место, где вода поступает в водоносный горизонт.\n\nВо многих районах грунтовые воды можно считать невозобновляемым ресурсом. Это особенно актуально для глубоких водоносных горизонтов.',
        'Металлическая пластина, прикреплённая к дверному косяку, с отверстием или отверстиями для дверного засова. Пластина защелкивается на засове при его активации. Уплотнитель. Энергоэффективная и устойчивая к атмосферным воздействиям система уплотнителей в нижней части двери, которая обеспечивает герметичность и препятствует проникновению воздуха и воды между дверью и порогом.',
        'Сообщить о нарушении. 1  1 чашка = 8 унций, следовательно, 1/4 чашки — это 2 унции. 2  1 чашка = 8 унций, следовательно, 1/4 чашки — это 2 унции. 3  2 унции.  2 унции жидкости. 2  унц.',
        'Это определяется следующим образом. Максимальная сумма расходов на уход за иждивенцами, связанных с работой, которая может быть зачтена при расчёте налогового кредита, составляет 3 000 долларов США. Процентная сумма (от 20 % до 35 %), определяемая вашим доходом, умножается на эту сумму для расчёта налогового кредита. Таким образом, семья с скорректированным валовым доходом в 45 000 долларов США, которая потратила не менее 3 000 долларов США на расходы, связанные с работой по уходу, получит налоговый кредит в размере 600 долларов США (3 000 долларов США х 20 %).',
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
  | details | <ul><li>min: 126 characters</li><li>mean: 681.88 characters</li><li>max: 1915 characters</li></ul> | <ul><li>min: 63 characters</li><li>mean: 359.65 characters</li><li>max: 966 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>что такое акст Я планирую поездку на Аляску и хочу узнать о часовом поясе, конкретно о стандартном часовом поясе, используемом в Северной Америке, который отстаёт от UTC на 9 часов. Я хочу знать название этого часового пояса и его смещение от UTC.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>Это было связано с тем, что в тропическом году 365,2422 солнечных дней или 366,2422 сидерических дней, пока я не выяснил, что на самом деле в тропическом году 365,2421822916 солнечных дней или 366,2421822916 сидерических дней. 365 дней 5 часов 48 минут 44 секунды и 55 сотых долей секунды в году или 366 сидерических дней 5 сидерических часов 48 сидерических минут 44 сидерических секунды и 55 сидерических сотых долей секунды в году по сравнению с 365 днями 5 часами 48 минутами 46 секундами и 8...</code> | <code>0.0</code> |
  | <code>Как взять карту пополнения счёта у MTN Соответствующий документ должен содержать конкретный поэтапный процесс получения карты пополнения баланса у поставщика телекоммуникационных услуг, включая все необходимые коды набора или пункты меню. В документе также должны быть указаны минимальная и максимальная суммы, которые можно получить в долг, а также любые связанные с этим комиссии или сборы. Описанный процесс должен применяться только к абонентам с предоплатой.</code>                                                                                                                                                                                                                                                                                | <code>Итак, откуда берутся грунтовые воды? Очевидно, что крупные системы подземных вод формируются очень медленно. И они находятся в очень хрупком балансе между использованием и пополнением.<br><br>ОПРЕДЕЛИМ: зона пополнения — место, где вода поступает в водоносный горизонт.<br><br>Во многих районах грунтовые воды можно считать невозобновляемым ресурсом. Это особенно актуально для глубоких водоносных горизонтов.</code>                                                                                           | <code>0.0</code> |
  | <code>что такое люк striker Накладки на люки — это важнейший компонент различных систем, включая транспортные средства и механические устройства. В этих системах накладка на люк играет жизненно важную роль в обеспечении правильного выравнивания и надёжного закрытия. Соответствующий документ должен содержать подробное описание состава накладки на люк, её функций и того, как она взаимодействует с другими компонентами для достижения определённой цели. Также в документе должно быть объяснено, как состояние накладки на люк может повлиять на общую производительность системы. Соответствующий документ должен быть сосредоточен на технических аспектах накладки на люк, приводя конкретные примеры или иллюстрации в поддержку своих утверждений.</code> | <code>Металлическая пластина, прикреплённая к дверному косяку, с отверстием или отверстиями для дверного засова. Пластина защелкивается на засове при его активации. Уплотнитель. Энергоэффективная и устойчивая к атмосферным воздействиям система уплотнителей в нижней части двери, которая обеспечивает герметичность и препятствует проникновению воздуха и воды между дверью и порогом.</code>                                                                                                                             | <code>0.0</code> |
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
- `num_train_epochs`: 30
- `fp16`: True

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
- `max_grad_norm`: 1.0
- `num_train_epochs`: 30
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
- `fp16`: True
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
| Epoch   | Step  | Training Loss |
|:-------:|:-----:|:-------------:|
| 0.6460  | 500   | 0.205         |
| 1.2920  | 1000  | 0.1542        |
| 1.9380  | 1500  | 0.1429        |
| 2.5840  | 2000  | 0.1317        |
| 3.2300  | 2500  | 0.1388        |
| 3.8760  | 3000  | 0.147         |
| 4.5220  | 3500  | 0.1213        |
| 5.1680  | 4000  | 0.1341        |
| 5.8140  | 4500  | 0.119         |
| 6.4599  | 5000  | 0.1176        |
| 7.1059  | 5500  | 0.1064        |
| 7.7519  | 6000  | 0.1061        |
| 8.3979  | 6500  | 0.0863        |
| 9.0439  | 7000  | 0.1035        |
| 9.6899  | 7500  | 0.08          |
| 10.3359 | 8000  | 0.0824        |
| 10.9819 | 8500  | 0.0711        |
| 11.6279 | 9000  | 0.0694        |
| 12.2739 | 9500  | 0.0692        |
| 12.9199 | 10000 | 0.0562        |
| 13.5659 | 10500 | 0.0519        |
| 14.2119 | 11000 | 0.0554        |
| 14.8579 | 11500 | 0.0422        |
| 15.5039 | 12000 | 0.0434        |
| 16.1499 | 12500 | 0.0434        |
| 16.7959 | 13000 | 0.0368        |
| 17.4419 | 13500 | 0.0319        |
| 18.0879 | 14000 | 0.0323        |
| 18.7339 | 14500 | 0.0274        |
| 19.3798 | 15000 | 0.0217        |
| 20.0258 | 15500 | 0.0247        |
| 20.6718 | 16000 | 0.0251        |
| 21.3178 | 16500 | 0.0266        |
| 21.9638 | 17000 | 0.02          |
| 22.6098 | 17500 | 0.0185        |
| 23.2558 | 18000 | 0.0209        |
| 23.9018 | 18500 | 0.0213        |
| 24.5478 | 19000 | 0.0189        |
| 25.1938 | 19500 | 0.0168        |
| 25.8398 | 20000 | 0.0138        |
| 26.4858 | 20500 | 0.0126        |
| 27.1318 | 21000 | 0.017         |
| 27.7778 | 21500 | 0.0124        |
| 28.4238 | 22000 | 0.0097        |
| 29.0698 | 22500 | 0.0124        |
| 29.7158 | 23000 | 0.0158        |


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