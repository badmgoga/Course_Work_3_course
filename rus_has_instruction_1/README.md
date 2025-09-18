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
    ['Какой актёр из «Ходячих мертвецов» скончался? Ходячие мертвецы — это популярный американский постапокалиптический хоррор-телесериал, основанный на одноимённой серии комиксов. В соответствующем документе должно быть чётко указано, что смерть или кончина конкретного актёра из сериала «Ходячие мертвецы» имела место, и дано ясное указание на то, что сообщённая смерть является ложной или мистификацией.', 'В этом умопомрачительном выпуске «Зомби-бум» мы тестируем тот самый арбалет, который использовал Дэрил Диксон из «Ходячих мертвецов». Мы также покажем вам, какие точки использовать для убийства зомби, если у вас будет выбор во время нашествия зомби.'],
    ['Какой актёр из «Ходячих мертвецов» скончался? Ходячие мертвецы — это популярный американский постапокалиптический хоррор-телесериал, основанный на одноимённой серии комиксов. В соответствующем документе должно быть чётко указано, что смерть или кончина конкретного актёра из сериала «Ходячие мертвецы» имела место, и дано ясное указание на то, что сообщённая смерть является ложной или мистификацией.', 'Телеактёр Майкл Лэндон умер в возрасте 54 лет: Голливуд. Звезда и продюсер нескольких успешных сериалов запомнился своим мужеством, юмором и характером. Майкл Лэндон, жизнерадостный и красивый актёр, чья жажда успеха привела его к зачастую смелой продюсерской карьере, умер в понедельник от рака, диагностированного три месяца назад.'],
    ['что такое тренинги по развитию чувствительности Документы, которые обсуждают чувствительность в контексте человеческих эмоций, социального поведения или личностного развития, имеют отношение к теме, в то время как те, которые сосредоточены на технических или научных измерениях чувствительности, — нет.', 'Краткое содержание урока. Систематическая десенсибилизация — это поведенческая методика, при которой человек постепенно подвергается воздействию объекта, события или места, вызывающих тревогу, одновременно занимаясь каким-либо видом релаксации, чтобы уменьшить симптомы тревоги. Например, очень распространённой фобией является страх перелёта.\n\nИерархия страхов. Следующий шаг в процессе систематической десенсибилизации включает в себя составление так называемой иерархии страхов — списка вещей, которые человек считает пугающими в связи с полётами, расположенных в порядке от наименее тревожных до наиболее тревожных.'],
    ['Причины неполного опорожнения кишечника Актуальный документ — это документ, который содержит конкретную практическую рекомендацию или решение для устранения проблемы неполного опорожнения кишечника и чётко объясняет механизм или причину, по которой предложенное решение эффективно. Документ также должен демонстрировать чёткое понимание физиологических процессов, связанных с дефекацией и опорожнением кишечника. Кроме того, документ не должен ограничиваться перечислением причин или симптомов неполного опорожнения кишечника, а должен предлагать реальный и практичный подход к смягчению или решению этой проблемы.', 'Изменение стула, диарея, частые дефекации, частые позывы к дефекации. Частые дефекации, частые позывы к дефекации, мышечные судороги или спазмы (болезненные), боль или дискомфорт. Запор, диарея, частые дефекации, боль или дискомфорт.'],
    ['с какой скоростью robocopy копирует данные Я системный администратор, мне поручено оптимизировать процессы передачи файлов в моей организации. Мне нужно найти информацию о скорости работы программы Robocopy при копировании данных. В частности, я ищу документы, которые содержат конкретные цифры или показатели производительности Robocopy. В документе также должно быть рассмотрено, какие преимущества даёт использование Robocopy по сравнению с другими методами передачи файлов. Я хочу узнать, может ли использование Robocopy значительно сократить время, необходимое для копирования больших объёмов данных.', 'Копирование и вставка таблиц из Bloomberg в Excel. Поместите курсор в угол таблицы в Bloomberg. Нажмите и перетащите данные, которые хотите скопировать. Bloomberg скопирует текст/данные автоматически без использования Ctrl + C. В Excel нажмите Ctrl + V, чтобы вставить данные. Щёлкните на меню «Данные» сверху на экране Excel и выберите «Текст по столбцам...».'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Какой актёр из «Ходячих мертвецов» скончался? Ходячие мертвецы — это популярный американский постапокалиптический хоррор-телесериал, основанный на одноимённой серии комиксов. В соответствующем документе должно быть чётко указано, что смерть или кончина конкретного актёра из сериала «Ходячие мертвецы» имела место, и дано ясное указание на то, что сообщённая смерть является ложной или мистификацией.',
    [
        'В этом умопомрачительном выпуске «Зомби-бум» мы тестируем тот самый арбалет, который использовал Дэрил Диксон из «Ходячих мертвецов». Мы также покажем вам, какие точки использовать для убийства зомби, если у вас будет выбор во время нашествия зомби.',
        'Телеактёр Майкл Лэндон умер в возрасте 54 лет: Голливуд. Звезда и продюсер нескольких успешных сериалов запомнился своим мужеством, юмором и характером. Майкл Лэндон, жизнерадостный и красивый актёр, чья жажда успеха привела его к зачастую смелой продюсерской карьере, умер в понедельник от рака, диагностированного три месяца назад.',
        'Краткое содержание урока. Систематическая десенсибилизация — это поведенческая методика, при которой человек постепенно подвергается воздействию объекта, события или места, вызывающих тревогу, одновременно занимаясь каким-либо видом релаксации, чтобы уменьшить симптомы тревоги. Например, очень распространённой фобией является страх перелёта.\n\nИерархия страхов. Следующий шаг в процессе систематической десенсибилизации включает в себя составление так называемой иерархии страхов — списка вещей, которые человек считает пугающими в связи с полётами, расположенных в порядке от наименее тревожных до наиболее тревожных.',
        'Изменение стула, диарея, частые дефекации, частые позывы к дефекации. Частые дефекации, частые позывы к дефекации, мышечные судороги или спазмы (болезненные), боль или дискомфорт. Запор, диарея, частые дефекации, боль или дискомфорт.',
        'Копирование и вставка таблиц из Bloomberg в Excel. Поместите курсор в угол таблицы в Bloomberg. Нажмите и перетащите данные, которые хотите скопировать. Bloomberg скопирует текст/данные автоматически без использования Ctrl + C. В Excel нажмите Ctrl + V, чтобы вставить данные. Щёлкните на меню «Данные» сверху на экране Excel и выберите «Текст по столбцам...».',
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
  | details | <ul><li>min: 126 characters</li><li>mean: 676.64 characters</li><li>max: 1915 characters</li></ul> | <ul><li>min: 66 characters</li><li>mean: 355.19 characters</li><li>max: 980 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                    | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Какой актёр из «Ходячих мертвецов» скончался? Ходячие мертвецы — это популярный американский постапокалиптический хоррор-телесериал, основанный на одноимённой серии комиксов. В соответствующем документе должно быть чётко указано, что смерть или кончина конкретного актёра из сериала «Ходячие мертвецы» имела место, и дано ясное указание на то, что сообщённая смерть является ложной или мистификацией.</code> | <code>В этом умопомрачительном выпуске «Зомби-бум» мы тестируем тот самый арбалет, который использовал Дэрил Диксон из «Ходячих мертвецов». Мы также покажем вам, какие точки использовать для убийства зомби, если у вас будет выбор во время нашествия зомби.</code>                                                                                                                                                                                                                                                                                                                                                                                         | <code>0.0</code> |
  | <code>Какой актёр из «Ходячих мертвецов» скончался? Ходячие мертвецы — это популярный американский постапокалиптический хоррор-телесериал, основанный на одноимённой серии комиксов. В соответствующем документе должно быть чётко указано, что смерть или кончина конкретного актёра из сериала «Ходячие мертвецы» имела место, и дано ясное указание на то, что сообщённая смерть является ложной или мистификацией.</code> | <code>Телеактёр Майкл Лэндон умер в возрасте 54 лет: Голливуд. Звезда и продюсер нескольких успешных сериалов запомнился своим мужеством, юмором и характером. Майкл Лэндон, жизнерадостный и красивый актёр, чья жажда успеха привела его к зачастую смелой продюсерской карьере, умер в понедельник от рака, диагностированного три месяца назад.</code>                                                                                                                                                                                                                                                                                                     | <code>0.0</code> |
  | <code>что такое тренинги по развитию чувствительности Документы, которые обсуждают чувствительность в контексте человеческих эмоций, социального поведения или личностного развития, имеют отношение к теме, в то время как те, которые сосредоточены на технических или научных измерениях чувствительности, — нет.</code>                                                                                                   | <code>Краткое содержание урока. Систематическая десенсибилизация — это поведенческая методика, при которой человек постепенно подвергается воздействию объекта, события или места, вызывающих тревогу, одновременно занимаясь каким-либо видом релаксации, чтобы уменьшить симптомы тревоги. Например, очень распространённой фобией является страх перелёта.<br><br>Иерархия страхов. Следующий шаг в процессе систематической десенсибилизации включает в себя составление так называемой иерархии страхов — списка вещей, которые человек считает пугающими в связи с полётами, расположенных в порядке от наименее тревожных до наиболее тревожных.</code> | <code>0.0</code> |
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
- `num_train_epochs`: 15
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
- `num_train_epochs`: 15
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
| 0.6460  | 500   | 0.2089        |
| 1.2920  | 1000  | 0.1587        |
| 1.9380  | 1500  | 0.1351        |
| 2.5840  | 2000  | 0.1519        |
| 3.2300  | 2500  | 0.1373        |
| 3.8760  | 3000  | 0.1377        |
| 4.5220  | 3500  | 0.1293        |
| 5.1680  | 4000  | 0.1382        |
| 5.8140  | 4500  | 0.132         |
| 6.4599  | 5000  | 0.1272        |
| 7.1059  | 5500  | 0.1167        |
| 7.7519  | 6000  | 0.1205        |
| 8.3979  | 6500  | 0.1103        |
| 9.0439  | 7000  | 0.0998        |
| 9.6899  | 7500  | 0.0984        |
| 10.3359 | 8000  | 0.0922        |
| 10.9819 | 8500  | 0.0916        |
| 11.6279 | 9000  | 0.0771        |
| 12.2739 | 9500  | 0.0899        |
| 12.9199 | 10000 | 0.0788        |
| 13.5659 | 10500 | 0.083         |
| 14.2119 | 11000 | 0.0672        |
| 14.8579 | 11500 | 0.0679        |


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