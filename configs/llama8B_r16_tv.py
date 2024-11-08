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
            # Path to model checkpoints stored locally
            # f'{MODEL_DIR}/llama/selected/snli.pt',
            # f'{MODEL_DIR}/llama/selected/mnli.pt',
            # f'{MODEL_DIR}/llama/selected/sick.pt',
            # f'{MODEL_DIR}/llama/selected/qnli.pt',
            # f'{MODEL_DIR}/llama/selected/rte.pt',
            # f'{MODEL_DIR}/llama/selected/scitail.pt',

            #HF models IDs
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_snli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_mnli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_sick',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_qnli',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_rte',
            'hoffman-lab/KnOTS-Llama3_8B_lora_R16_scitail',
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
        'merge_method': 'tv',
        'merging_type': 'sum',
        'merge_other_params' : False,
        'scaling_coeffs': [.5],
    },
    'eval_type': 'logits',
}

