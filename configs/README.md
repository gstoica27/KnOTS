# Configs
Configs define our experimental test suites.
The name of each config file describes the (1) model, (2) LoRA rank, (3) merging method used in an experiment.
We codify these as "<model-name>_r<lora-rank>_<merging-method>.py", although this is purely a subjective choice.
You can title the config files however you wish, and reference them accordingly in the experiment scripts.

## Fields
Each config looks like this: 
```python
import os

VIT_ARCH = 'ViT-B-32-CLIP'                               # Model Architecture
MODEL_DIR = ''                                           # Model Directory
CACHE_DIR = ''                                           # Where to cache HF pretrained checkpoints
HEAD_DIR = ''                                            # CLIP Head Directory

config = {
    'dataset': [{                                        # Specifies datasets used
        'name': "<DATASET_NAME>",                        # name of the dataset. Should match a corresponding variable name found in datasets/config.py
        'shuffle_train': True,                           # Whether to shuffle train set
        'crop_ratio': 1.0                                # Image crop ratio
        'clip_encodings': ""                             # Path to CLIP head, obtained from dataset/parsing/generate_clip_heads.py
        'val_fraction': .2                               # Proportion of test set to subsample as validation (remainder left for test). Only used when no validation data exists.
        'batch_size': 32,                                # Batch size
        'num_workers': 16,                               # Number of workers to use
        'shuffled_idxs': os.path.join(                   # Path to indices used to split test into a validation and new test set.
          os.getcwd(),
          'dataset/shuffled_idxs/<DATASET_NAME>_shuffled_idxs.pt'
        )
    }, ...],
    'model': {                                           # Specifies types of models used
        'name': 'hf_clip',                               # Model type
        'base_type': "openai/clip-vit-base-patch32",     # HF name
        'cachedir': CACHE_DIR,                           # Directory where HF downloads the pretrained model checkpoint
        'bases': [
          f'{MODEL_DIR}/<DATASET_NAME>_lora.pt', ...     # Paths to LoRA models. The order should match Dataset order
    ],
    'ft_config': {                                        # Specifies FT setup
        'type': 'lora',                                   # FT type
        'r': 16,                                          # LoRA Rank used
        'lora_alpha': 16,                                 # Layers LoRA is applied over
        'target_modules': [
          "q_proj", "k_proj",
          "v_proj", "out_proj"
        ],
        'lora_dropout': 0.1,                              # Dropout on LoRA
        'bias': "none",                                   # Whether to include bias on LoRA models
    },
    'task_merge_config': {                                # Specifies merging method configuration. In this case, it is KnOTS-TIES.
        'representation': 'svd-vector',                   # Representation for merging. Can be "vector" (standard for TA, TIES, DARE) or "svd-vector" (KnOTS).
        'sign_resolve_mode': 'sum_of_values',             # How to resolve sign conflict in TIES. Can either be "sum_of_values" (sum of raw values) or "sum_of_signs" (sum of signs of values).
        'scaling_coeffs': .6, #[.6],                      # Scaling coefficient for merging.
        'topK': 20,                                       # TopK value from TIES.
        'merge_method': 'ties',                           # Specifies we merge with TIES.
        'merging_type': 'mean',                           # Merging models with mean. Can be any from {"sum", "max", "unmerged" (no merging is performed)}. All operations are elementwise.
        'concat_across_output': True,                     # Whether to concatenate along the columns or rows.
        'dare' : False,                                   # Whether to apply DARE
        'dare_pruning_coeffs' : 0.0,                      # Pruning coefficients with DARE when activated.
    },
    'eval_type': 'clip'                                   # Specifies that we merge a "clip" model.
}
```

## New Configs
Adding new configs can be done simply by following the above format. 
