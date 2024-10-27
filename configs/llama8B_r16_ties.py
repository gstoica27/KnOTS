CACHE_DIR = ''          # Path to the cache directory
MODEL_DIR = ''          # Path to the model directory
INGREDIENTS_PATH = ""   # Path to the ingredients file (If exists)
PTM_PATH = ""           # Path to the pre-trained model

config = {
    'dataset': [
        {
            'name': 'snli',
            'mask_class': None,
        },

        {
            'name': 'mnli',
            'val_fraction': 0.2,
            'mask_class': None,
        },
        {
            'name': 'sick',
            'mask_class': None,
        },
        {
            'name': 'qnli',
            'val_fraction': 0.2,
            'mask_class': 1,
        },
        {
            'name': 'rte',
            'val_fraction': 0.5,
            'mask_class': 1,
        },
        {
            'name': 'scitail',
            'mask_class': 2,
        },
    
    ],
    'model': {
        'name' : 'meta-llama/Meta-Llama-3-8B',
        'ptm_path': PTM_PATH,
        'cachedir': CACHE_DIR,
        'bases': [
            f'{MODEL_DIR}/llama/selected/snli.pt',
            f'{MODEL_DIR}/llama/selected/mnli.pt',
            f'{MODEL_DIR}/llama/selected/sick.pt',
            f'{MODEL_DIR}/llama/selected/qnli.pt',
            f'{MODEL_DIR}/llama/selected/rte.pt',
            f'{MODEL_DIR}/llama/selected/scitail.pt',

        ],
        'ft_config': {
            'type': 'lora',
            'subtype': 'peft',
            },


        'peft_config': {
            'task_type' : "SEQ_CLS",
            'inference_mode' : False,
            'r': 16,
            'lora_alpha' : 16,
            'lora_dropout' : 0.1,
            'target_modules' : ["q_proj", "k_proj", "v_proj", "o_proj"]
        },
    },
    'task_merge_config': {
        'ingredients_path' : INGREDIENTS_PATH,
        'representation': 'vector',
        'sign_resolve_mode': 'sum_of_values',
        'topK': 100,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'scaling_coeffs': [.5],
    },
    'eval_type': 'logits',
}

