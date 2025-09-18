---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:12431
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
    ['сердце расположено там, где ', 'Сортировка: по новизне, по старшинству. Лучший ответ: странное название, не могу найти сердце. Сердце расположено в грудной полости, в части груди, известной как средостение — область между плевральными мешками, где содержатся все грудные внутренние органы, кроме лёгких. Перикард — это тонкий мешок или мембрана, в котором находится сердце и корни крупных кровеносных сосудов. Когда вы упоминаете о полости тела, вы говорите, что сердце находится в грудной клетке или грудной полости. В этой полости оно лежит в средостении и окружено перикардом.'],
    ['где расположен эпицентр землетрясения ', 'Землетрясение длилось 20 секунд.\n\nГДЕ: Эпицентр находился на разломе Сан-Андреас примерно в 89,5 км к югу от Сан-Франциско и в 16 км к северо-востоку от Санта-Крус, недалеко от горы Лома-Приета в горах Санта-Крус. Глубина очага составила 17,7 км (типичная глубина очагов землетрясений в Калифорнии — от 6,4 до 9,6 км).\n\n20-летие землетрясения в Лома-Приета 1989 года служит важным напоминанием для всех жителей Калифорнии о том, что геологические процессы, ответственные за создание прекрасного природного ландшафта, которым мы наслаждаемся, иногда могут происходить внезапно и с разрушительной силой, поэтому важно быть готовыми.'],
    ['значение имени mpho ', 'Ирландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.\n\nАнглийское значение: имя Кин (Keane) — английское имя для ребёнка. На английском языке значение имени Кин: острый.\n\nКельтское значение: имя Кин (Keane) — кельтское имя для ребёнка. На кельтском языке значение имени Кин: высокий и красивый.\n\nИрландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.\n\nАнглийское значение: имя Кин (Keane) — английское имя для ребёнка.'],
    ['Можешь ли ты купить мелатонин в Великобритании? ', 'Шишковидная железа содержит большое количество кальция, и с её помощью рентгенологи могут отметить середину мозга на рентгеновских снимках.\n\nЧем занимается шишковидная железа? Шишковидная железа наиболее известна тем, что выделяет гормон мелатонин, который поступает в кровь и, возможно, также в спинномозговую жидкость.\n\nСуточные (циркадные) биологические часы организма контролируют выработку мелатонина шишковидной железой, поэтому мелатонин обычно используется в исследованиях человека для понимания биологических ритмов организма.'],
    ['С какой скоростью может бежать Супермен в милях в час? ', 'С самого начала Fox News сообщает, что Болт достигает скорости 28 миль в час. BBC объясняет, как он достигает такой скорости: мировое рекордное время Болта 2009 года: Болт преодолел дистанцию со старта с места на отметке 23,35 мили в час. Однако от отметки 60 метров до 80 метров он пробежал за 1,61 секунды, что примерно соответствует скорости 27,79 мили в час.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'сердце расположено там, где ',
    [
        'Сортировка: по новизне, по старшинству. Лучший ответ: странное название, не могу найти сердце. Сердце расположено в грудной полости, в части груди, известной как средостение — область между плевральными мешками, где содержатся все грудные внутренние органы, кроме лёгких. Перикард — это тонкий мешок или мембрана, в котором находится сердце и корни крупных кровеносных сосудов. Когда вы упоминаете о полости тела, вы говорите, что сердце находится в грудной клетке или грудной полости. В этой полости оно лежит в средостении и окружено перикардом.',
        'Землетрясение длилось 20 секунд.\n\nГДЕ: Эпицентр находился на разломе Сан-Андреас примерно в 89,5 км к югу от Сан-Франциско и в 16 км к северо-востоку от Санта-Крус, недалеко от горы Лома-Приета в горах Санта-Крус. Глубина очага составила 17,7 км (типичная глубина очагов землетрясений в Калифорнии — от 6,4 до 9,6 км).\n\n20-летие землетрясения в Лома-Приета 1989 года служит важным напоминанием для всех жителей Калифорнии о том, что геологические процессы, ответственные за создание прекрасного природного ландшафта, которым мы наслаждаемся, иногда могут происходить внезапно и с разрушительной силой, поэтому важно быть готовыми.',
        'Ирландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.\n\nАнглийское значение: имя Кин (Keane) — английское имя для ребёнка. На английском языке значение имени Кин: острый.\n\nКельтское значение: имя Кин (Keane) — кельтское имя для ребёнка. На кельтском языке значение имени Кин: высокий и красивый.\n\nИрландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.\n\nАнглийское значение: имя Кин (Keane) — английское имя для ребёнка.',
        'Шишковидная железа содержит большое количество кальция, и с её помощью рентгенологи могут отметить середину мозга на рентгеновских снимках.\n\nЧем занимается шишковидная железа? Шишковидная железа наиболее известна тем, что выделяет гормон мелатонин, который поступает в кровь и, возможно, также в спинномозговую жидкость.\n\nСуточные (циркадные) биологические часы организма контролируют выработку мелатонина шишковидной железой, поэтому мелатонин обычно используется в исследованиях человека для понимания биологических ритмов организма.',
        'С самого начала Fox News сообщает, что Болт достигает скорости 28 миль в час. BBC объясняет, как он достигает такой скорости: мировое рекордное время Болта 2009 года: Болт преодолел дистанцию со старта с места на отметке 23,35 мили в час. Однако от отметки 60 метров до 80 метров он пробежал за 1,61 секунды, что примерно соответствует скорости 27,79 мили в час.',
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

* Size: 12,431 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                        | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                            | float                                                          |
  | details | <ul><li>min: 13 characters</li><li>mean: 41.89 characters</li><li>max: 151 characters</li></ul> | <ul><li>min: 36 characters</li><li>mean: 365.64 characters</li><li>max: 1022 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.04</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | label            |
  |:----------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>сердце расположено там, где </code>           | <code>Сортировка: по новизне, по старшинству. Лучший ответ: странное название, не могу найти сердце. Сердце расположено в грудной полости, в части груди, известной как средостение — область между плевральными мешками, где содержатся все грудные внутренние органы, кроме лёгких. Перикард — это тонкий мешок или мембрана, в котором находится сердце и корни крупных кровеносных сосудов. Когда вы упоминаете о полости тела, вы говорите, что сердце находится в грудной клетке или грудной полости. В этой полости оно лежит в средостении и окружено перикардом.</code>                                                                                                | <code>0.0</code> |
  | <code>где расположен эпицентр землетрясения </code> | <code>Землетрясение длилось 20 секунд.<br><br>ГДЕ: Эпицентр находился на разломе Сан-Андреас примерно в 89,5 км к югу от Сан-Франциско и в 16 км к северо-востоку от Санта-Крус, недалеко от горы Лома-Приета в горах Санта-Крус. Глубина очага составила 17,7 км (типичная глубина очагов землетрясений в Калифорнии — от 6,4 до 9,6 км).<br><br>20-летие землетрясения в Лома-Приета 1989 года служит важным напоминанием для всех жителей Калифорнии о том, что геологические процессы, ответственные за создание прекрасного природного ландшафта, которым мы наслаждаемся, иногда могут происходить внезапно и с разрушительной силой, поэтому важно быть готовыми.</code> | <code>0.0</code> |
  | <code>значение имени mpho </code>                   | <code>Ирландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.<br><br>Английское значение: имя Кин (Keane) — английское имя для ребёнка. На английском языке значение имени Кин: острый.<br><br>Кельтское значение: имя Кин (Keane) — кельтское имя для ребёнка. На кельтском языке значение имени Кин: высокий и красивый.<br><br>Ирландское значение: имя Кин (Keane) — ирландское имя для ребёнка. На ирландском языке значение имени Кин: древний.<br><br>Английское значение: имя Кин (Keane) — английское имя для ребёнка.</code>                                                                              | <code>0.0</code> |
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
| 0.6435 | 500  | 0.4384        |
| 1.2870 | 1000 | 0.144         |
| 1.9305 | 1500 | 0.1375        |
| 2.5740 | 2000 | 0.1265        |
| 3.2175 | 2500 | 0.1316        |
| 3.8610 | 3000 | 0.1251        |
| 4.5045 | 3500 | 0.1234        |
| 5.1480 | 4000 | 0.1144        |
| 5.7915 | 4500 | 0.1165        |
| 6.4350 | 5000 | 0.1137        |
| 7.0785 | 5500 | 0.1067        |
| 7.7220 | 6000 | 0.1049        |
| 8.3655 | 6500 | 0.0984        |
| 9.0090 | 7000 | 0.0964        |
| 9.6525 | 7500 | 0.0894        |


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