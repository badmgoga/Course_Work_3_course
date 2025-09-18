---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:310000
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
    ['what is personal exemption', 'Personal exemption (United States) - Phase-out. 1  The personal exemptions begin to phase out when AGI exceeds $305,050 for 2014 joint tax returns and $254,200 for 2014 single tax returns.'],
    ["what is average salary for front desk manager? A relevant document is one that provides a specific numerical value or range of values for the average salary of a front desk manager in a particular region or country, and also provides a credible source or methodology for the salary estimate. The document should explicitly mention the job title 'front desk manager' and provide a direct answer to the query. The salary information should be the main focus of the document, and not be mentioned in passing or as part of a broader discussion. The document should also be written in a formal and informative tone, and should not contain personal opinions or anecdotes.", 'From millions of real job salary data. 6 Hotel Front Office Manager salary data. Average Hotel Front Office Manager salary is $57,949 Detailed Hotel Front Office Manager starting salary, median salary, pay scale, bonus data report Register & Know how much $ you can earn | Sign In Home (current)'],
    ['what situations are you allowed to ignore posted traffic regulations and signals? Traffic regulations and signals are an essential part of ensuring safety on the roads. In many countries, traffic laws and regulations are in place to govern the behavior of drivers, pedestrians, and cyclists. These regulations can vary greatly from country to country, and even from region to region within a country. For instance, some countries drive on the left side of the road, while others drive on the right side. Additionally, traffic signals and signs can have different meanings depending on the context in which they are used. When evaluating the relevance of a document to this query, a relevant document must provide specific information about situations in which traffic regulations and signals can be ignored or disobeyed, and must provide concrete examples or explanations to support its claims. The document should also demonstrate a clear understanding of the nuances of traffic regulations and signals, and should not simply provide general information about traffic laws or regulations. Furthermore, a relevant document should not focus primarily on the consequences of disobeying traffic regulations, but rather on the specific circumstances under which it is permissible to do so.', 'OFAC regulations often provide general licenses authorizing the performance of certain categories of transactions. OFAC also issues specific licenses on a case-by-case basis under certain limited situations and conditions. Guidance on how to request a specific license is found below and at 31 C.F.R. 501.801.'],
    ['what is oklahoma state income tax rate', 'For the latter, if income from your business passes through to you personally, that income will be subject to taxation on your personal state tax return. Marylandâ\x80\x99s corporation income tax is assessed at a flat rate of 8.25% of net income.'],
    ["where is the radar for the weather? I'm a meteorology student researching how weather radar systems work, and I need information on the technical aspects of radar systems used for weather forecasting, specifically how they detect and display weather patterns. A relevant document should provide detailed explanations of radar technology and its applications in meteorology.", 'Composite weather radar echoes and precipitation forecasts up to 60 minutes ahead are displayed in 1 km x 1 km resolution every 5 minutes, respectively. Any out-of-operation radars may cause radar echoes in affected areas to be weaker than they should be or not displayed at all.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'what is personal exemption',
    [
        'Personal exemption (United States) - Phase-out. 1  The personal exemptions begin to phase out when AGI exceeds $305,050 for 2014 joint tax returns and $254,200 for 2014 single tax returns.',
        'From millions of real job salary data. 6 Hotel Front Office Manager salary data. Average Hotel Front Office Manager salary is $57,949 Detailed Hotel Front Office Manager starting salary, median salary, pay scale, bonus data report Register & Know how much $ you can earn | Sign In Home (current)',
        'OFAC regulations often provide general licenses authorizing the performance of certain categories of transactions. OFAC also issues specific licenses on a case-by-case basis under certain limited situations and conditions. Guidance on how to request a specific license is found below and at 31 C.F.R. 501.801.',
        'For the latter, if income from your business passes through to you personally, that income will be subject to taxation on your personal state tax return. Marylandâ\x80\x99s corporation income tax is assessed at a flat rate of 8.25% of net income.',
        'Composite weather radar echoes and precipitation forecasts up to 60 minutes ahead are displayed in 1 km x 1 km resolution every 5 minutes, respectively. Any out-of-operation radars may cause radar echoes in affected areas to be weaker than they should be or not displayed at all.',
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

* Size: 310,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                        | sentence_1                                                                                      | label                                                          |
  |:--------|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                            | string                                                                                          | float                                                          |
  | details | <ul><li>min: 12 characters</li><li>mean: 375.06 characters</li><li>max: 1825 characters</li></ul> | <ul><li>min: 59 characters</li><li>mean: 345.2 characters</li><li>max: 850 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                                         | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what is personal exemption</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | <code>Personal exemption (United States) - Phase-out. 1  The personal exemptions begin to phase out when AGI exceeds $305,050 for 2014 joint tax returns and $254,200 for 2014 single tax returns.</code>                                                                                                                          | <code>0.0</code> |
  | <code>what is average salary for front desk manager? A relevant document is one that provides a specific numerical value or range of values for the average salary of a front desk manager in a particular region or country, and also provides a credible source or methodology for the salary estimate. The document should explicitly mention the job title 'front desk manager' and provide a direct answer to the query. The salary information should be the main focus of the document, and not be mentioned in passing or as part of a broader discussion. The document should also be written in a formal and informative tone, and should not contain personal opinions or anecdotes.</code>                                                                                                                                                                                                                                                                                                                                                   | <code>From millions of real job salary data. 6 Hotel Front Office Manager salary data. Average Hotel Front Office Manager salary is $57,949 Detailed Hotel Front Office Manager starting salary, median salary, pay scale, bonus data report Register & Know how much $ you can earn \| Sign In Home (current)</code>              | <code>0.0</code> |
  | <code>what situations are you allowed to ignore posted traffic regulations and signals? Traffic regulations and signals are an essential part of ensuring safety on the roads. In many countries, traffic laws and regulations are in place to govern the behavior of drivers, pedestrians, and cyclists. These regulations can vary greatly from country to country, and even from region to region within a country. For instance, some countries drive on the left side of the road, while others drive on the right side. Additionally, traffic signals and signs can have different meanings depending on the context in which they are used. When evaluating the relevance of a document to this query, a relevant document must provide specific information about situations in which traffic regulations and signals can be ignored or disobeyed, and must provide concrete examples or explanations to support its claims. The document should also demonstrate a clear understanding of the nuances of traffic regulations and sign...</code> | <code>OFAC regulations often provide general licenses authorizing the performance of certain categories of transactions. OFAC also issues specific licenses on a case-by-case basis under certain limited situations and conditions. Guidance on how to request a specific license is found below and at 31 C.F.R. 501.801.</code> | <code>0.0</code> |
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
- `num_train_epochs`: 3
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
<details><summary>Click to expand</summary>

| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0258 | 500   | 0.1465        |
| 0.0516 | 1000  | 0.0854        |
| 0.0774 | 1500  | 0.0847        |
| 0.1032 | 2000  | 0.0783        |
| 0.1290 | 2500  | 0.0799        |
| 0.1548 | 3000  | 0.0761        |
| 0.1806 | 3500  | 0.0763        |
| 0.2065 | 4000  | 0.0755        |
| 0.2323 | 4500  | 0.0757        |
| 0.2581 | 5000  | 0.0731        |
| 0.2839 | 5500  | 0.0762        |
| 0.3097 | 6000  | 0.0673        |
| 0.3355 | 6500  | 0.0648        |
| 0.3613 | 7000  | 0.0749        |
| 0.3871 | 7500  | 0.0712        |
| 0.4129 | 8000  | 0.0688        |
| 0.4387 | 8500  | 0.0677        |
| 0.4645 | 9000  | 0.0637        |
| 0.4903 | 9500  | 0.063         |
| 0.5161 | 10000 | 0.0694        |
| 0.5419 | 10500 | 0.059         |
| 0.5677 | 11000 | 0.06          |
| 0.5935 | 11500 | 0.0629        |
| 0.6194 | 12000 | 0.0697        |
| 0.6452 | 12500 | 0.0686        |
| 0.6710 | 13000 | 0.0649        |
| 0.6968 | 13500 | 0.0686        |
| 0.7226 | 14000 | 0.0665        |
| 0.7484 | 14500 | 0.0681        |
| 0.7742 | 15000 | 0.0633        |
| 0.8    | 15500 | 0.0646        |
| 0.8258 | 16000 | 0.0618        |
| 0.8516 | 16500 | 0.0566        |
| 0.8774 | 17000 | 0.0648        |
| 0.9032 | 17500 | 0.0683        |
| 0.9290 | 18000 | 0.0668        |
| 0.9548 | 18500 | 0.0597        |
| 0.9806 | 19000 | 0.0656        |
| 1.0065 | 19500 | 0.059         |
| 1.0323 | 20000 | 0.0517        |
| 1.0581 | 20500 | 0.0536        |
| 1.0839 | 21000 | 0.0463        |
| 1.1097 | 21500 | 0.0545        |
| 1.1355 | 22000 | 0.0533        |
| 1.1613 | 22500 | 0.0532        |
| 1.1871 | 23000 | 0.0513        |
| 1.2129 | 23500 | 0.0562        |
| 1.2387 | 24000 | 0.0521        |
| 1.2645 | 24500 | 0.048         |
| 1.2903 | 25000 | 0.0543        |
| 1.3161 | 25500 | 0.0496        |
| 1.3419 | 26000 | 0.0392        |
| 1.3677 | 26500 | 0.0603        |
| 1.3935 | 27000 | 0.0466        |
| 1.4194 | 27500 | 0.0584        |
| 1.4452 | 28000 | 0.0548        |
| 1.4710 | 28500 | 0.0559        |
| 1.4968 | 29000 | 0.0551        |
| 1.5226 | 29500 | 0.0514        |
| 1.5484 | 30000 | 0.0463        |
| 1.5742 | 30500 | 0.0526        |
| 1.6    | 31000 | 0.0513        |
| 1.6258 | 31500 | 0.0508        |
| 1.6516 | 32000 | 0.0511        |
| 1.6774 | 32500 | 0.0498        |
| 1.7032 | 33000 | 0.0528        |
| 1.7290 | 33500 | 0.0443        |
| 1.7548 | 34000 | 0.049         |
| 1.7806 | 34500 | 0.0486        |
| 1.8065 | 35000 | 0.0559        |
| 1.8323 | 35500 | 0.0574        |
| 1.8581 | 36000 | 0.0447        |
| 1.8839 | 36500 | 0.0452        |
| 1.9097 | 37000 | 0.0548        |
| 1.9355 | 37500 | 0.047         |
| 1.9613 | 38000 | 0.0449        |
| 1.9871 | 38500 | 0.0505        |
| 2.0129 | 39000 | 0.0411        |
| 2.0387 | 39500 | 0.0351        |
| 2.0645 | 40000 | 0.0397        |
| 2.0903 | 40500 | 0.0369        |
| 2.1161 | 41000 | 0.0353        |
| 2.1419 | 41500 | 0.0414        |
| 2.1677 | 42000 | 0.0389        |
| 2.1935 | 42500 | 0.0358        |
| 2.2194 | 43000 | 0.0389        |
| 2.2452 | 43500 | 0.0362        |
| 2.2710 | 44000 | 0.0345        |
| 2.2968 | 44500 | 0.0362        |
| 2.3226 | 45000 | 0.0378        |
| 2.3484 | 45500 | 0.0461        |
| 2.3742 | 46000 | 0.0347        |
| 2.4    | 46500 | 0.046         |
| 2.4258 | 47000 | 0.0382        |
| 2.4516 | 47500 | 0.0449        |
| 2.4774 | 48000 | 0.0401        |
| 2.5032 | 48500 | 0.0372        |
| 2.5290 | 49000 | 0.0417        |
| 2.5548 | 49500 | 0.0417        |
| 2.5806 | 50000 | 0.0377        |
| 2.6065 | 50500 | 0.04          |
| 2.6323 | 51000 | 0.0382        |
| 2.6581 | 51500 | 0.0454        |
| 2.6839 | 52000 | 0.0355        |
| 2.7097 | 52500 | 0.04          |
| 2.7355 | 53000 | 0.0371        |
| 2.7613 | 53500 | 0.0336        |
| 2.7871 | 54000 | 0.0352        |
| 2.8129 | 54500 | 0.0444        |
| 2.8387 | 55000 | 0.035         |
| 2.8645 | 55500 | 0.0371        |
| 2.8903 | 56000 | 0.0339        |
| 2.9161 | 56500 | 0.038         |
| 2.9419 | 57000 | 0.0409        |
| 2.9677 | 57500 | 0.0373        |
| 2.9935 | 58000 | 0.0325        |

</details>

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