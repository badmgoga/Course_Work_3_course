---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:309442
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
    ["when was new amsterdam founded? I'm a historian researching the early colonial history of North America, and I'm particularly interested in understanding the founding of specific settlements and colonies. I'm looking for documents that provide a clear and direct answer to my question, with a specific date and a clear description of the event or circumstances surrounding the founding. The document should also clearly identify the entity or organization responsible for the founding, such as a country, company, or individual. I'm not interested in documents that only mention the settlement or colony in passing, or those that provide vague or indirect information about its founding. I also want to exclude documents that focus on the modern-day characteristics or prices of the settlement or colony, as that's not relevant to my research. Please provide me with documents that meet these criteria, as they will be invaluable in helping me understand this important period in American history.", 'The Dutch East India Company was founded in 1602 and remained active until 1799. The Dutch name was Vereenigde Oost-Indische Compagnie, what literally means the United East Indian Company.'],
    ["how much does cost to level yard? As a homeowner planning to regrade my yard, I'm looking for information on the average cost of this project. I want to know the typical price range that homeowners pay to have a professional do the work, including any variations in cost depending on the size of the yard or the complexity of the job. A relevant document should provide a specific cost estimate or range, and ideally include information on the factors that affect the final cost. I'm not interested in general advice or tips on yard maintenance, and I don't care about the cost of other landscaping projects.", '1 The minimum cost of concrete is approximately $70 per cubic yard. 2  The maximum cost of concrete is approximately $80 per cubic yard.  The square footage covered by a cubic yard of concrete varies based on the depth of the feature to be poured.'],
    ['would cisco aps act as a wireless client? Wireless access points can operate in various modes, including infrastructure mode and ad-hoc mode. A document is relevant if it describes a specific scenario where a Cisco AP acts as a wireless client, and provides technical details about the configuration or functionality in this scenario.', "Cheep AP's connected to the same home router will have ARP issues jumping between APs. Higher end APs have something called AP roaming or Fast AP roaming for jumping from AP to AP without drooping link. Fast roaming is required for like wireless IPphones. http://www.techworld.com/mobile/wlan-roaming--the-basics-435/."],
    ["prednisone is prescribed for a client with diabetes mellitus who is taking nph insulin daily. which prescription should the nurse anticipate during therapy with the prednisone? As a nurse preparing for a pharmacology exam, I need to find information on how to adjust a patient's insulin dosage when they are prescribed prednisone, specifically looking for scenarios where the patient is already taking a specific type of insulin. A relevant document should provide a clear answer to this question and mention the type of insulin the patient is taking.", '1 A group of autoimmune disorders in which the immune system attacks and destroys blood vessels. 2  Type I diabetes mellitus. 3  Appears to be caused by an antibody that attacks and destroys the islet cells of the pancreas, which produce insulin. A group of autoimmune disorders in which the immune system attacks and destroys blood vessels. 2  Type I diabetes mellitus. 3  Appears to be caused by an antibody that attacks and destroys the islet cells of the pancreas, which produce insulin.'],
    ["what's in a moscow mule? A relevant document is one that provides a direct answer to the query, explicitly stating the ingredients or components of a Moscow Mule. It should not be a recipe or a set of directions, but rather a descriptive passage that explains what a Moscow Mule is. Documents that discuss other types of drinks or substances, or those that provide unrelated information about Russia or copper, are not relevant.", "Gin Gin Mule. gin, lime juice, mint, ginger beer. The Gin Gin Mule is a nice little twist on the Moscow Mule that, well, uses gin instead of vodka. Gin and ginger beer have a long history as favorites in the UK, so it was only natural that they'd eventually end up together in this delicious variation."],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "when was new amsterdam founded? I'm a historian researching the early colonial history of North America, and I'm particularly interested in understanding the founding of specific settlements and colonies. I'm looking for documents that provide a clear and direct answer to my question, with a specific date and a clear description of the event or circumstances surrounding the founding. The document should also clearly identify the entity or organization responsible for the founding, such as a country, company, or individual. I'm not interested in documents that only mention the settlement or colony in passing, or those that provide vague or indirect information about its founding. I also want to exclude documents that focus on the modern-day characteristics or prices of the settlement or colony, as that's not relevant to my research. Please provide me with documents that meet these criteria, as they will be invaluable in helping me understand this important period in American history.",
    [
        'The Dutch East India Company was founded in 1602 and remained active until 1799. The Dutch name was Vereenigde Oost-Indische Compagnie, what literally means the United East Indian Company.',
        '1 The minimum cost of concrete is approximately $70 per cubic yard. 2  The maximum cost of concrete is approximately $80 per cubic yard.  The square footage covered by a cubic yard of concrete varies based on the depth of the feature to be poured.',
        "Cheep AP's connected to the same home router will have ARP issues jumping between APs. Higher end APs have something called AP roaming or Fast AP roaming for jumping from AP to AP without drooping link. Fast roaming is required for like wireless IPphones. http://www.techworld.com/mobile/wlan-roaming--the-basics-435/.",
        '1 A group of autoimmune disorders in which the immune system attacks and destroys blood vessels. 2  Type I diabetes mellitus. 3  Appears to be caused by an antibody that attacks and destroys the islet cells of the pancreas, which produce insulin. A group of autoimmune disorders in which the immune system attacks and destroys blood vessels. 2  Type I diabetes mellitus. 3  Appears to be caused by an antibody that attacks and destroys the islet cells of the pancreas, which produce insulin.',
        "Gin Gin Mule. gin, lime juice, mint, ginger beer. The Gin Gin Mule is a nice little twist on the Moscow Mule that, well, uses gin instead of vodka. Gin and ginger beer have a long history as favorites in the UK, so it was only natural that they'd eventually end up together in this delicious variation.",
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

* Size: 309,442 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                         | sentence_1                                                                                      | label                                                          |
  |:--------|:---------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                             | string                                                                                          | float                                                          |
  | details | <ul><li>min: 180 characters</li><li>mean: 689.02 characters</li><li>max: 1788 characters</li></ul> | <ul><li>min: 57 characters</li><li>mean: 340.2 characters</li><li>max: 882 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.04</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                  | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>when was new amsterdam founded? I'm a historian researching the early colonial history of North America, and I'm particularly interested in understanding the founding of specific settlements and colonies. I'm looking for documents that provide a clear and direct answer to my question, with a specific date and a clear description of the event or circumstances surrounding the founding. The document should also clearly identify the entity or organization responsible for the founding, such as a country, company, or individual. I'm not interested in documents that only mention the settlement or colony in passing, or those that provide vague or indirect information about its founding. I also want to exclude documents that focus on the modern-day characteristics or prices of the settlement or colony, as that's not relevant to my research. Please provide me with documents that meet these criteria, as they will be invaluable in helping me understand this important period in American history.</code> | <code>The Dutch East India Company was founded in 1602 and remained active until 1799. The Dutch name was Vereenigde Oost-Indische Compagnie, what literally means the United East Indian Company.</code>                                                                                                                                   | <code>0.0</code> |
  | <code>how much does cost to level yard? As a homeowner planning to regrade my yard, I'm looking for information on the average cost of this project. I want to know the typical price range that homeowners pay to have a professional do the work, including any variations in cost depending on the size of the yard or the complexity of the job. A relevant document should provide a specific cost estimate or range, and ideally include information on the factors that affect the final cost. I'm not interested in general advice or tips on yard maintenance, and I don't care about the cost of other landscaping projects.</code>                                                                                                                                                                                                                                                                                                                                                                                                      | <code>1 The minimum cost of concrete is approximately $70 per cubic yard. 2  The maximum cost of concrete is approximately $80 per cubic yard.  The square footage covered by a cubic yard of concrete varies based on the depth of the feature to be poured.</code>                                                                        | <code>0.0</code> |
  | <code>would cisco aps act as a wireless client? Wireless access points can operate in various modes, including infrastructure mode and ad-hoc mode. A document is relevant if it describes a specific scenario where a Cisco AP acts as a wireless client, and provides technical details about the configuration or functionality in this scenario.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>Cheep AP's connected to the same home router will have ARP issues jumping between APs. Higher end APs have something called AP roaming or Fast AP roaming for jumping from AP to AP without drooping link. Fast roaming is required for like wireless IPphones. http://www.techworld.com/mobile/wlan-roaming--the-basics-435/.</code> | <code>0.0</code> |
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
| 0.0259 | 500   | 0.0986        |
| 0.0517 | 1000  | 0.0845        |
| 0.0776 | 1500  | 0.0707        |
| 0.1034 | 2000  | 0.0668        |
| 0.1293 | 2500  | 0.0731        |
| 0.1551 | 3000  | 0.0722        |
| 0.1810 | 3500  | 0.0707        |
| 0.2068 | 4000  | 0.0673        |
| 0.2327 | 4500  | 0.0625        |
| 0.2585 | 5000  | 0.0675        |
| 0.2844 | 5500  | 0.0672        |
| 0.3102 | 6000  | 0.0586        |
| 0.3361 | 6500  | 0.0618        |
| 0.3619 | 7000  | 0.0609        |
| 0.3878 | 7500  | 0.0616        |
| 0.4136 | 8000  | 0.0576        |
| 0.4395 | 8500  | 0.0617        |
| 0.4653 | 9000  | 0.0637        |
| 0.4912 | 9500  | 0.0557        |
| 0.5170 | 10000 | 0.0544        |
| 0.5429 | 10500 | 0.0638        |
| 0.5687 | 11000 | 0.0622        |
| 0.5946 | 11500 | 0.0486        |
| 0.6204 | 12000 | 0.061         |
| 0.6463 | 12500 | 0.0585        |
| 0.6721 | 13000 | 0.0516        |
| 0.6980 | 13500 | 0.0552        |
| 0.7239 | 14000 | 0.0527        |
| 0.7497 | 14500 | 0.0532        |
| 0.7756 | 15000 | 0.053         |
| 0.8014 | 15500 | 0.05          |
| 0.8273 | 16000 | 0.0626        |
| 0.8531 | 16500 | 0.0523        |
| 0.8790 | 17000 | 0.0505        |
| 0.9048 | 17500 | 0.0537        |
| 0.9307 | 18000 | 0.0578        |
| 0.9565 | 18500 | 0.0498        |
| 0.9824 | 19000 | 0.053         |
| 1.0082 | 19500 | 0.0492        |
| 1.0341 | 20000 | 0.0385        |
| 1.0599 | 20500 | 0.0318        |
| 1.0858 | 21000 | 0.0402        |
| 1.1116 | 21500 | 0.0379        |
| 1.1375 | 22000 | 0.0376        |
| 1.1633 | 22500 | 0.0452        |
| 1.1892 | 23000 | 0.0411        |
| 1.2150 | 23500 | 0.0451        |
| 1.2409 | 24000 | 0.0458        |
| 1.2667 | 24500 | 0.0411        |
| 1.2926 | 25000 | 0.0338        |
| 1.3184 | 25500 | 0.0442        |
| 1.3443 | 26000 | 0.0455        |
| 1.3701 | 26500 | 0.0409        |
| 1.3960 | 27000 | 0.0395        |
| 1.4218 | 27500 | 0.0427        |
| 1.4477 | 28000 | 0.0412        |
| 1.4736 | 28500 | 0.0451        |
| 1.4994 | 29000 | 0.0419        |
| 1.5253 | 29500 | 0.0389        |
| 1.5511 | 30000 | 0.0372        |
| 1.5770 | 30500 | 0.0388        |
| 1.6028 | 31000 | 0.047         |
| 1.6287 | 31500 | 0.0422        |
| 1.6545 | 32000 | 0.0336        |
| 1.6804 | 32500 | 0.0478        |
| 1.7062 | 33000 | 0.0472        |
| 1.7321 | 33500 | 0.0416        |
| 1.7579 | 34000 | 0.045         |
| 1.7838 | 34500 | 0.0408        |
| 1.8096 | 35000 | 0.047         |
| 1.8355 | 35500 | 0.0467        |
| 1.8613 | 36000 | 0.0362        |
| 1.8872 | 36500 | 0.0458        |
| 1.9130 | 37000 | 0.0404        |
| 1.9389 | 37500 | 0.0407        |
| 1.9647 | 38000 | 0.038         |
| 1.9906 | 38500 | 0.0357        |
| 2.0164 | 39000 | 0.0321        |
| 2.0423 | 39500 | 0.0304        |
| 2.0681 | 40000 | 0.0246        |
| 2.0940 | 40500 | 0.0339        |
| 2.1198 | 41000 | 0.0328        |
| 2.1457 | 41500 | 0.0247        |
| 2.1716 | 42000 | 0.0274        |
| 2.1974 | 42500 | 0.0251        |
| 2.2233 | 43000 | 0.0309        |
| 2.2491 | 43500 | 0.0312        |
| 2.2750 | 44000 | 0.029         |
| 2.3008 | 44500 | 0.0211        |
| 2.3267 | 45000 | 0.0324        |
| 2.3525 | 45500 | 0.0305        |
| 2.3784 | 46000 | 0.0262        |
| 2.4042 | 46500 | 0.0263        |
| 2.4301 | 47000 | 0.0279        |
| 2.4559 | 47500 | 0.0325        |
| 2.4818 | 48000 | 0.0313        |
| 2.5076 | 48500 | 0.0268        |
| 2.5335 | 49000 | 0.0232        |
| 2.5593 | 49500 | 0.0307        |
| 2.5852 | 50000 | 0.0208        |
| 2.6110 | 50500 | 0.0244        |
| 2.6369 | 51000 | 0.0277        |
| 2.6627 | 51500 | 0.0293        |
| 2.6886 | 52000 | 0.0273        |
| 2.7144 | 52500 | 0.0282        |
| 2.7403 | 53000 | 0.0314        |
| 2.7661 | 53500 | 0.0269        |
| 2.7920 | 54000 | 0.0311        |
| 2.8178 | 54500 | 0.0285        |
| 2.8437 | 55000 | 0.0248        |
| 2.8696 | 55500 | 0.0238        |
| 2.8954 | 56000 | 0.0369        |
| 2.9213 | 56500 | 0.0322        |
| 2.9471 | 57000 | 0.0242        |
| 2.9730 | 57500 | 0.0304        |
| 2.9988 | 58000 | 0.024         |

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