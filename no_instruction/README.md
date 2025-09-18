---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:310558
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
    ['what actor died this past week', "Nov 11, 2016 1:33 PM EST. Robert Vaughn, an actor with a long and varied career best known for '60s spy series The Man from U.N.C.L.E., has died. Deadline reports that Vaughn died on Friday after a brief battle with leukemia. He was 83. Vaughn played suave spy Napoleon Solo,â\x80¦ Read more."],
    ['divisions of the cerebral hemispheres that are named after the overlying skull bones', 'Each of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.Smaller commissures, including the anterior commissure, the posterior commissure and the hippocampal commissure also join the hemispheres and these are also present in other vertebrates.ach of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.'],
    ['how do the protein requirements change for pregnancy', 'Rate This Article : 1 2 3 4 5. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. Pregnancy is an experience of growth, change, enrichment and challenge. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. 1  Pregnancy is an experience of growth, change, enrichment and challenge.  Women experience many emotions during pregnancy, starting with the first trimester.'],
    ['how many people go on diets each year', 'The average American now eats roughly 193 pounds of beef, pork and/or chicken a year (or more than 3.7 pounds a week), up from roughly 184 pounds in 2012. Among the reasons: A stronger dollar and large increases in the supply of chicken and pork, says William Sawyer, the director of food and agricultural research at Rabobank.'],
    ['how much does a hmmwv weigh', '1 At breeding time, heifers should weigh at least 65-70 percent of their mature body weight. 2  At calving time, they need to weigh at least 85 percent of their mature body weight. From weaning to calving, a heifer should be gaining at a rate of 1.5 to 1.75 lbs/day. 2  At breeding time, heifers should weigh at least 65-70 percent of their mature body weight.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'what actor died this past week',
    [
        "Nov 11, 2016 1:33 PM EST. Robert Vaughn, an actor with a long and varied career best known for '60s spy series The Man from U.N.C.L.E., has died. Deadline reports that Vaughn died on Friday after a brief battle with leukemia. He was 83. Vaughn played suave spy Napoleon Solo,â\x80¦ Read more.",
        'Each of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.Smaller commissures, including the anterior commissure, the posterior commissure and the hippocampal commissure also join the hemispheres and these are also present in other vertebrates.ach of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.',
        'Rate This Article : 1 2 3 4 5. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. Pregnancy is an experience of growth, change, enrichment and challenge. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. 1  Pregnancy is an experience of growth, change, enrichment and challenge.  Women experience many emotions during pregnancy, starting with the first trimester.',
        'The average American now eats roughly 193 pounds of beef, pork and/or chicken a year (or more than 3.7 pounds a week), up from roughly 184 pounds in 2012. Among the reasons: A stronger dollar and large increases in the supply of chicken and pork, says William Sawyer, the director of food and agricultural research at Rabobank.',
        '1 At breeding time, heifers should weigh at least 65-70 percent of their mature body weight. 2  At calving time, they need to weigh at least 85 percent of their mature body weight. From weaning to calving, a heifer should be gaining at a rate of 1.5 to 1.75 lbs/day. 2  At breeding time, heifers should weigh at least 65-70 percent of their mature body weight.',
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

* Size: 310,558 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                                      | sentence_1                                                                                        | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                            | float                                                          |
  | details | <ul><li>min: 10 characters</li><li>mean: 32.81 characters</li><li>max: 176 characters</li></ul> | <ul><li>min: 65 characters</li><li>mean: 346.23 characters</li><li>max: 1023 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.03</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                        | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | label            |
  |:--------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>what actor died this past week</code>                                                       | <code>Nov 11, 2016 1:33 PM EST. Robert Vaughn, an actor with a long and varied career best known for '60s spy series The Man from U.N.C.L.E., has died. Deadline reports that Vaughn died on Friday after a brief battle with leukemia. He was 83. Vaughn played suave spy Napoleon Solo,â¦ Read more.</code>                                                                                                                                                                                                                                                                                                                                                                                                                         | <code>0.0</code> |
  | <code>divisions of the cerebral hemispheres that are named after the overlying skull bones</code> | <code>Each of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.Smaller commissures, including the anterior commissure, the posterior commissure and the hippocampal commissure also join the hemispheres and these are also present in other vertebrates.ach of these hemispheres has an outer layer of grey matter, the cerebral cortex, that is supported by an inner layer of white matter. In eutherian (placental) mammals, the hemispheres are linked by the corpus callosum, a very large bundle of nerve fibers.</code> | <code>0.0</code> |
  | <code>how do the protein requirements change for pregnancy</code>                                 | <code>Rate This Article : 1 2 3 4 5. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. Pregnancy is an experience of growth, change, enrichment and challenge. During the 40 weeks of pregnancy, the expectant mother will go through several physical and emotional changes. 1  Pregnancy is an experience of growth, change, enrichment and challenge.  Women experience many emotions during pregnancy, starting with the first trimester.</code>                                                                                                                                                                                                                      | <code>0.0</code> |
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
| 0.0258 | 500   | 0.1718        |
| 0.0515 | 1000  | 0.0899        |
| 0.0773 | 1500  | 0.0817        |
| 0.1030 | 2000  | 0.0817        |
| 0.1288 | 2500  | 0.0769        |
| 0.1546 | 3000  | 0.0772        |
| 0.1803 | 3500  | 0.0681        |
| 0.2061 | 4000  | 0.0708        |
| 0.2318 | 4500  | 0.0777        |
| 0.2576 | 5000  | 0.0735        |
| 0.2834 | 5500  | 0.0726        |
| 0.3091 | 6000  | 0.0816        |
| 0.3349 | 6500  | 0.0704        |
| 0.3606 | 7000  | 0.0733        |
| 0.3864 | 7500  | 0.077         |
| 0.4122 | 8000  | 0.0671        |
| 0.4379 | 8500  | 0.0811        |
| 0.4637 | 9000  | 0.069         |
| 0.4894 | 9500  | 0.0724        |
| 0.5152 | 10000 | 0.0724        |
| 0.5410 | 10500 | 0.0665        |
| 0.5667 | 11000 | 0.0706        |
| 0.5925 | 11500 | 0.0698        |
| 0.6182 | 12000 | 0.0665        |
| 0.6440 | 12500 | 0.0771        |
| 0.6698 | 13000 | 0.0789        |
| 0.6955 | 13500 | 0.0705        |
| 0.7213 | 14000 | 0.0712        |
| 0.7470 | 14500 | 0.0684        |
| 0.7728 | 15000 | 0.0784        |
| 0.7986 | 15500 | 0.0689        |
| 0.8243 | 16000 | 0.0683        |
| 0.8501 | 16500 | 0.0684        |
| 0.8758 | 17000 | 0.0632        |
| 0.9016 | 17500 | 0.0665        |
| 0.9274 | 18000 | 0.067         |
| 0.9531 | 18500 | 0.0679        |
| 0.9789 | 19000 | 0.0694        |
| 1.0046 | 19500 | 0.066         |
| 1.0304 | 20000 | 0.0623        |
| 1.0562 | 20500 | 0.0548        |
| 1.0819 | 21000 | 0.0561        |
| 1.1077 | 21500 | 0.0549        |
| 1.1334 | 22000 | 0.0601        |
| 1.1592 | 22500 | 0.0547        |
| 1.1850 | 23000 | 0.0615        |
| 1.2107 | 23500 | 0.0533        |
| 1.2365 | 24000 | 0.0602        |
| 1.2622 | 24500 | 0.0631        |
| 1.2880 | 25000 | 0.0598        |
| 1.3138 | 25500 | 0.0588        |
| 1.3395 | 26000 | 0.0628        |
| 1.3653 | 26500 | 0.0571        |
| 1.3910 | 27000 | 0.0624        |
| 1.4168 | 27500 | 0.0636        |
| 1.4426 | 28000 | 0.0587        |
| 1.4683 | 28500 | 0.0494        |
| 1.4941 | 29000 | 0.0578        |
| 1.5198 | 29500 | 0.054         |
| 1.5456 | 30000 | 0.0621        |
| 1.5714 | 30500 | 0.0656        |
| 1.5971 | 31000 | 0.0574        |
| 1.6229 | 31500 | 0.0567        |
| 1.6486 | 32000 | 0.0544        |
| 1.6744 | 32500 | 0.0593        |
| 1.7002 | 33000 | 0.0624        |
| 1.7259 | 33500 | 0.0593        |
| 1.7517 | 34000 | 0.0542        |
| 1.7774 | 34500 | 0.0546        |
| 1.8032 | 35000 | 0.0637        |
| 1.8290 | 35500 | 0.0581        |
| 1.8547 | 36000 | 0.0605        |
| 1.8805 | 36500 | 0.0606        |
| 1.9062 | 37000 | 0.0583        |
| 1.9320 | 37500 | 0.0602        |
| 1.9578 | 38000 | 0.064         |
| 1.9835 | 38500 | 0.0588        |
| 2.0093 | 39000 | 0.0536        |
| 2.0350 | 39500 | 0.0444        |
| 2.0608 | 40000 | 0.043         |
| 2.0866 | 40500 | 0.0445        |
| 2.1123 | 41000 | 0.0518        |
| 2.1381 | 41500 | 0.0484        |
| 2.1638 | 42000 | 0.038         |
| 2.1896 | 42500 | 0.0432        |
| 2.2154 | 43000 | 0.0461        |
| 2.2411 | 43500 | 0.0524        |
| 2.2669 | 44000 | 0.0444        |
| 2.2926 | 44500 | 0.0432        |
| 2.3184 | 45000 | 0.053         |
| 2.3442 | 45500 | 0.0461        |
| 2.3699 | 46000 | 0.046         |
| 2.3957 | 46500 | 0.0546        |
| 2.4214 | 47000 | 0.0475        |
| 2.4472 | 47500 | 0.0475        |
| 2.4730 | 48000 | 0.0448        |
| 2.4987 | 48500 | 0.0581        |
| 2.5245 | 49000 | 0.0492        |
| 2.5502 | 49500 | 0.0433        |
| 2.5760 | 50000 | 0.0456        |
| 2.6018 | 50500 | 0.0536        |
| 2.6275 | 51000 | 0.0431        |
| 2.6533 | 51500 | 0.0522        |
| 2.6790 | 52000 | 0.0545        |
| 2.7048 | 52500 | 0.0517        |
| 2.7306 | 53000 | 0.0487        |
| 2.7563 | 53500 | 0.0518        |
| 2.7821 | 54000 | 0.0449        |
| 2.8078 | 54500 | 0.0455        |
| 2.8336 | 55000 | 0.0517        |
| 2.8594 | 55500 | 0.0421        |
| 2.8851 | 56000 | 0.0484        |
| 2.9109 | 56500 | 0.0388        |
| 2.9366 | 57000 | 0.0462        |
| 2.9624 | 57500 | 0.0406        |
| 2.9882 | 58000 | 0.0522        |

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