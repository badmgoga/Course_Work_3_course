---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:31000
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
    ["what's the difference between a townhouse and a condo? When evaluating the relevance of a document to this query, consider the following criteria: A relevant document must provide a clear and concise explanation of the differences between townhouses and condos, specifically highlighting the distinct aspects of ownership and property rights associated with each type of dwelling. The document should demonstrate a thorough understanding of the legal and practical implications of these differences, and provide concrete examples or scenarios to illustrate the concepts. Furthermore, a relevant document should maintain a formal and informative tone, avoiding colloquial language, slang, and personal opinions. It should also be free of irrelevant information, such as personal anecdotes, unrelated topics, or unnecessary details. A document that meets these criteria will be considered relevant to the query. Evaluate the provided documents based on these standards to determine their relevance.In addition to the above criteria, pay particular attention to the level of specificity and detail provided in the document. A relevant document should provide more than just a general overview of the topic, but rather delve into the nuances and intricacies of the differences between townhouses and condos. It should also demonstrate an ability to distinguish between the two types of dwellings, highlighting the unique characteristics and features of each. A document that fails to meet these standards, or provides incomplete, inaccurate, or misleading information, will be considered non-relevant to the query.", '3042 DISSTON St is a townhouse in PHILADELPHIA, PA 19149. This 1,174 square foot townhouse sits on a 1,825 square foot lot and features 3 bedrooms and 1.5 bathrooms. This townhouse has been listed on Redfin since February 20, 2017 and is currently priced at $174,000. This property was built in 1950.'],
    ['fastest thing is.? A relevant document is one that discusses the fastest thing in a specific domain or category, such as the fastest speed achieved by a particular object or entity, and provides a specific measurement or quantifiable value to support this claim. The document should also be written in a formal or informative tone, suggesting that the information is factual and reliable. Documents that only provide a list of fastest things or entities without providing additional context or specific measurements are not relevant. Additionally, documents that focus on the fastest person or athlete are also not relevant.', 'The Cheetah is a large cat and is known for being the fastest of all the land animals. It can reach speeds of up to 70 miles per hour. The Cheetah can accelerate from 0 to 60 in around 3 1/2 seconds. This is faster than most sports cars!'],
    ['causes of incomplete bowel emptying', 'Soy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.oy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.'],
    ['how are hairdressers paid', 'Hairdressers, Hairstylists, and Cosmetologists will usually earn a wage of Twenty Six Thousand Five Hundred dollars yearly. Hairdressers, Hairstylists, and Cosmetologists receive the highest salary in the District of Columbia, where they can get salary pay of close to $43330. Employees with these job titles are paid at the highest level in Public Administration, where they can get pay of $31750.'],
    ['zenker definition? The esophagus is a muscular tube that carries food and liquids from the throat to the stomach. It is a vital part of the digestive system and is susceptible to various conditions and disorders. When searching for information on a specific medical term, it is essential to prioritize documents that provide a clear and concise definition, accompanied by relevant anatomical details. A relevant document should explicitly state the definition of the term and describe its location in the body. Furthermore, it should provide information on the causes or characteristics of the condition.', "A Zenker's diverticulum, also pharyngoesophageal diverticulum, also pharyngeal pouch, also hypopharyngeal diverticulum, is a diverticulum of the mucosa of the pharynx, just above the cricopharyngeal muscle (i.e. above the upper sphincter of the esophagus).n simple words, when there is excessive pressure within the lower pharynx, the weakest portion of the pharyngeal wall balloons out, forming a diverticulum which may reach several centimetres in diameter."],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "what's the difference between a townhouse and a condo? When evaluating the relevance of a document to this query, consider the following criteria: A relevant document must provide a clear and concise explanation of the differences between townhouses and condos, specifically highlighting the distinct aspects of ownership and property rights associated with each type of dwelling. The document should demonstrate a thorough understanding of the legal and practical implications of these differences, and provide concrete examples or scenarios to illustrate the concepts. Furthermore, a relevant document should maintain a formal and informative tone, avoiding colloquial language, slang, and personal opinions. It should also be free of irrelevant information, such as personal anecdotes, unrelated topics, or unnecessary details. A document that meets these criteria will be considered relevant to the query. Evaluate the provided documents based on these standards to determine their relevance.In addition to the above criteria, pay particular attention to the level of specificity and detail provided in the document. A relevant document should provide more than just a general overview of the topic, but rather delve into the nuances and intricacies of the differences between townhouses and condos. It should also demonstrate an ability to distinguish between the two types of dwellings, highlighting the unique characteristics and features of each. A document that fails to meet these standards, or provides incomplete, inaccurate, or misleading information, will be considered non-relevant to the query.",
    [
        '3042 DISSTON St is a townhouse in PHILADELPHIA, PA 19149. This 1,174 square foot townhouse sits on a 1,825 square foot lot and features 3 bedrooms and 1.5 bathrooms. This townhouse has been listed on Redfin since February 20, 2017 and is currently priced at $174,000. This property was built in 1950.',
        'The Cheetah is a large cat and is known for being the fastest of all the land animals. It can reach speeds of up to 70 miles per hour. The Cheetah can accelerate from 0 to 60 in around 3 1/2 seconds. This is faster than most sports cars!',
        'Soy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.oy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.',
        'Hairdressers, Hairstylists, and Cosmetologists will usually earn a wage of Twenty Six Thousand Five Hundred dollars yearly. Hairdressers, Hairstylists, and Cosmetologists receive the highest salary in the District of Columbia, where they can get salary pay of close to $43330. Employees with these job titles are paid at the highest level in Public Administration, where they can get pay of $31750.',
        "A Zenker's diverticulum, also pharyngoesophageal diverticulum, also pharyngeal pouch, also hypopharyngeal diverticulum, is a diverticulum of the mucosa of the pharynx, just above the cricopharyngeal muscle (i.e. above the upper sphincter of the esophagus).n simple words, when there is excessive pressure within the lower pharynx, the weakest portion of the pharyngeal wall balloons out, forming a diverticulum which may reach several centimetres in diameter.",
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

* Size: 31,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                       | sentence_1                                                                                       | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                           | string                                                                                           | float                                                          |
  | details | <ul><li>min: 9 characters</li><li>mean: 359.22 characters</li><li>max: 1710 characters</li></ul> | <ul><li>min: 55 characters</li><li>mean: 344.08 characters</li><li>max: 906 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what's the difference between a townhouse and a condo? When evaluating the relevance of a document to this query, consider the following criteria: A relevant document must provide a clear and concise explanation of the differences between townhouses and condos, specifically highlighting the distinct aspects of ownership and property rights associated with each type of dwelling. The document should demonstrate a thorough understanding of the legal and practical implications of these differences, and provide concrete examples or scenarios to illustrate the concepts. Furthermore, a relevant document should maintain a formal and informative tone, avoiding colloquial language, slang, and personal opinions. It should also be free of irrelevant information, such as personal anecdotes, unrelated topics, or unnecessary details. A document that meets these criteria will be considered relevant to the query. Evaluate the provided documents based on these standards to determine their relevance.In a...</code> | <code>3042 DISSTON St is a townhouse in PHILADELPHIA, PA 19149. This 1,174 square foot townhouse sits on a 1,825 square foot lot and features 3 bedrooms and 1.5 bathrooms. This townhouse has been listed on Redfin since February 20, 2017 and is currently priced at $174,000. This property was built in 1950.</code>                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>fastest thing is.? A relevant document is one that discusses the fastest thing in a specific domain or category, such as the fastest speed achieved by a particular object or entity, and provides a specific measurement or quantifiable value to support this claim. The document should also be written in a formal or informative tone, suggesting that the information is factual and reliable. Documents that only provide a list of fastest things or entities without providing additional context or specific measurements are not relevant. Additionally, documents that focus on the fastest person or athlete are also not relevant.</code>                                                                                                                                                                                                                                                                                                                                                                                            | <code>The Cheetah is a large cat and is known for being the fastest of all the land animals. It can reach speeds of up to 70 miles per hour. The Cheetah can accelerate from 0 to 60 in around 3 1/2 seconds. This is faster than most sports cars!</code>                                                                                                                                                                                                                                                                                                                                         | <code>0.0</code> |
  | <code>causes of incomplete bowel emptying</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>Soy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.oy is the only non-animal protein that is complete. Incomplete proteins are found in non-animal foods. These foods are generally considered healthy and should be included in a balanced diet. Examples of foods high in incomplete protein include nuts, beans, legumes, rice and grains.</code> | <code>0.0</code> |
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
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.2580 | 500  | 0.1455        |
| 0.5160 | 1000 | 0.0912        |
| 0.7740 | 1500 | 0.0851        |
| 1.0320 | 2000 | 0.0709        |
| 1.2900 | 2500 | 0.0639        |
| 1.5480 | 3000 | 0.0494        |
| 1.8060 | 3500 | 0.0578        |
| 2.0640 | 4000 | 0.0555        |
| 2.3220 | 4500 | 0.0383        |
| 2.5800 | 5000 | 0.0461        |
| 2.8380 | 5500 | 0.0419        |


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