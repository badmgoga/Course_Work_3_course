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
    ['заболевания, вызывающие чрезмерное газообразование ', 'Причины избыточного газообразования. Избыточное газообразование (метеоризм) обычно вызвано заглатыванием воздуха во время еды или питья. Также продукты с высоким содержанием клетчатки, такие как бобовые и капуста, и газированные напитки могут вызывать избыточное газообразование.'],
    ['какая часть почки более тёмная, красно-коричневого цвета ', 'Мозговое вещество почки обычно более тёмно-красно-коричневого цвета по сравнению с корковым веществом. Расположенное глубже в почке, мозговое вещество состоит из почечных пирамид, которые отвечают за концентрацию мочи. Более тёмный цвет этой области обусловлен более высокой плотностью кровеносных сосудов и канальцев, которые играют решающую роль в функции почек по фильтрации и выведению продуктов жизнедеятельности.'],
    ['как сделать копию экрана на компьютере ', 'Проверьте информацию об операционной системе в Windows 8.1 или Windows RT 8.1. Проведите пальцем от правого края экрана, коснитесь «Параметры», а затем коснитесь «Изменение параметров ПК». (Если вы используете мышь, наведите указатель мыши в правый нижний угол экрана, переместите указатель мыши вверх, щёлкните «Параметры», а затем щёлкните «Изменение параметров ПК».)'],
    ['зачем был нужен трансатлантический кабель ', 'Коаксиальный кабель — это разновидность медного кабеля, но он гораздо лучше подходит для передачи высокоскоростных данных, чем витые пары проводов в сети ADSL. Фактически, коаксиальный кабель, который в настоящее время используют компании Telstra и Optus, станет частью финальной инфраструктуры NBN.'],
    ['определение космоса ', 'Как неделимое целое, как система человек также является частью больших целостностей или систем — семьи, сообщества, всего человечества, нашей планеты и космоса. В этих контекстах мы сталкиваемся с тремя важными жизненными задачами: занятие (деятельность), любовь и секс, а также наши отношения с другими людьми — все это социальные вызовы.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'заболевания, вызывающие чрезмерное газообразование ',
    [
        'Причины избыточного газообразования. Избыточное газообразование (метеоризм) обычно вызвано заглатыванием воздуха во время еды или питья. Также продукты с высоким содержанием клетчатки, такие как бобовые и капуста, и газированные напитки могут вызывать избыточное газообразование.',
        'Мозговое вещество почки обычно более тёмно-красно-коричневого цвета по сравнению с корковым веществом. Расположенное глубже в почке, мозговое вещество состоит из почечных пирамид, которые отвечают за концентрацию мочи. Более тёмный цвет этой области обусловлен более высокой плотностью кровеносных сосудов и канальцев, которые играют решающую роль в функции почек по фильтрации и выведению продуктов жизнедеятельности.',
        'Проверьте информацию об операционной системе в Windows 8.1 или Windows RT 8.1. Проведите пальцем от правого края экрана, коснитесь «Параметры», а затем коснитесь «Изменение параметров ПК». (Если вы используете мышь, наведите указатель мыши в правый нижний угол экрана, переместите указатель мыши вверх, щёлкните «Параметры», а затем щёлкните «Изменение параметров ПК».)',
        'Коаксиальный кабель — это разновидность медного кабеля, но он гораздо лучше подходит для передачи высокоскоростных данных, чем витые пары проводов в сети ADSL. Фактически, коаксиальный кабель, который в настоящее время используют компании Telstra и Optus, станет частью финальной инфраструктуры NBN.',
        'Как неделимое целое, как система человек также является частью больших целостностей или систем — семьи, сообщества, всего человечества, нашей планеты и космоса. В этих контекстах мы сталкиваемся с тремя важными жизненными задачами: занятие (деятельность), любовь и секс, а также наши отношения с другими людьми — все это социальные вызовы.',
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
  | details | <ul><li>min: 13 characters</li><li>mean: 41.03 characters</li><li>max: 151 characters</li></ul> | <ul><li>min: 88 characters</li><li>mean: 360.71 characters</li><li>max: 1019 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                      | label            |
  |:-----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>заболевания, вызывающие чрезмерное газообразование </code>       | <code>Причины избыточного газообразования. Избыточное газообразование (метеоризм) обычно вызвано заглатыванием воздуха во время еды или питья. Также продукты с высоким содержанием клетчатки, такие как бобовые и капуста, и газированные напитки могут вызывать избыточное газообразование.</code>                                                                                                                                            | <code>0.0</code> |
  | <code>какая часть почки более тёмная, красно-коричневого цвета </code> | <code>Мозговое вещество почки обычно более тёмно-красно-коричневого цвета по сравнению с корковым веществом. Расположенное глубже в почке, мозговое вещество состоит из почечных пирамид, которые отвечают за концентрацию мочи. Более тёмный цвет этой области обусловлен более высокой плотностью кровеносных сосудов и канальцев, которые играют решающую роль в функции почек по фильтрации и выведению продуктов жизнедеятельности.</code> | <code>1.0</code> |
  | <code>как сделать копию экрана на компьютере </code>                   | <code>Проверьте информацию об операционной системе в Windows 8.1 или Windows RT 8.1. Проведите пальцем от правого края экрана, коснитесь «Параметры», а затем коснитесь «Изменение параметров ПК». (Если вы используете мышь, наведите указатель мыши в правый нижний угол экрана, переместите указатель мыши вверх, щёлкните «Параметры», а затем щёлкните «Изменение параметров ПК».)</code>                                                  | <code>0.0</code> |
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
| 0.6435  | 500   | 0.5748        |
| 1.2870  | 1000  | 0.1423        |
| 1.9305  | 1500  | 0.14          |
| 2.5740  | 2000  | 0.1352        |
| 3.2175  | 2500  | 0.1289        |
| 3.8610  | 3000  | 0.1306        |
| 4.5045  | 3500  | 0.117         |
| 5.1480  | 4000  | 0.123         |
| 5.7915  | 4500  | 0.1206        |
| 6.4350  | 5000  | 0.1063        |
| 7.0785  | 5500  | 0.1116        |
| 7.7220  | 6000  | 0.0991        |
| 8.3655  | 6500  | 0.0958        |
| 9.0090  | 7000  | 0.0974        |
| 9.6525  | 7500  | 0.0939        |
| 10.2960 | 8000  | 0.0652        |
| 10.9395 | 8500  | 0.0827        |
| 11.5830 | 9000  | 0.067         |
| 12.2265 | 9500  | 0.0645        |
| 12.8700 | 10000 | 0.0633        |
| 13.5135 | 10500 | 0.0542        |
| 14.1570 | 11000 | 0.0609        |
| 14.8005 | 11500 | 0.0522        |
| 15.4440 | 12000 | 0.0453        |
| 16.0875 | 12500 | 0.0465        |
| 16.7310 | 13000 | 0.0339        |
| 17.3745 | 13500 | 0.0348        |
| 18.0180 | 14000 | 0.0375        |
| 18.6615 | 14500 | 0.0311        |
| 19.3050 | 15000 | 0.0217        |
| 19.9485 | 15500 | 0.0262        |
| 20.5920 | 16000 | 0.0238        |
| 21.2355 | 16500 | 0.0208        |
| 21.8790 | 17000 | 0.0169        |
| 22.5225 | 17500 | 0.0175        |
| 23.1660 | 18000 | 0.0179        |
| 23.8095 | 18500 | 0.0182        |
| 24.4530 | 19000 | 0.0178        |
| 25.0965 | 19500 | 0.011         |
| 25.7400 | 20000 | 0.0188        |
| 26.3835 | 20500 | 0.014         |
| 27.0270 | 21000 | 0.0073        |
| 27.6705 | 21500 | 0.0128        |
| 28.3140 | 22000 | 0.0108        |
| 28.9575 | 22500 | 0.01          |
| 29.6010 | 23000 | 0.0116        |


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