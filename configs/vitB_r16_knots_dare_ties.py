import os

VIT_ARCH = 'ViT-B-32-CLIP'  # Model Architecture
MODEL_DIR = ''              # Model Directory
CACHE_DIR = ''              # Where to cache HF pretrained checkpoints
HEAD_DIR = ''               # CLIP Head Directory

config = {
    'dataset': [
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/cars_shuffled_idxs.pt')
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'eurosat',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt')
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,  
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt')
        },
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'sun397',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'sun397_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/sun397_shuffled_idxs.pt')
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': "openai/clip-vit-base-patch32",
        'cachedir': CACHE_DIR,
        'bases': [
            # Path to model checkpoints stored locally - rank-16 8 vision tasks
            # f'{MODEL_DIR}/stanford_cars_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/dtd_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/eurosat_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/gtsrb_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/mnist_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/resisc45_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/sun397_lora-r16-alpha16-dropout0.1.pt',
            # f'{MODEL_DIR}/svhn_lora-r16-alpha16-dropout0.1.pt',

            #HF models IDs
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_dtd',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_eurosat',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_gtsrb',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_resisc45',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_sun397',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_svhn',
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    
    'task_merge_config': {
        'representation': 'svd-vector',
        'sign_resolve_mode': 'sum_of_values',
        'scaling_coeffs': .7, #[.6],
        'topK': 100,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'concat_across_output': True,   # When True: U is of size O x nI; False: U is of size I x nO
        'dare' : True,
        'dare_pruning_coeffs' : 0.1,
        'dare_seed': 421,
    },
    'eval_type': 'clip'
}

