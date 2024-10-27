import pdb
import os
import torch
import torchvision
import random
import numpy as np
import sys
from utils import *
from transformers import CLIPProcessor, CLIPModel
from dataset.templates import get_templates
from tqdm import tqdm


def build_classification_head(model, tokenizer, classnames, template, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            embeddings = []
            for t in template:
                tokenized_template = tokenizer(t(classname))
                tokenized_template = {k: torch.tensor(v).to(device).reshape(1, -1) for k, v in tokenized_template.items()}
                embedding = model.text_projection(model.text_model(**tokenized_template)[1])
                embeddings.append(embedding)
            embeddings = torch.concat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return zeroshot_weights


if __name__ == '__main__':
    # Uncomment one of these depending on model you want.
    vit_path = "openai/clip-vit-large-patch14"
    # vit_path = "openai/clip-vit-base-patch32"
    cache_dir = ""                              # Path to HF cache directory
    classification_heads_dir = ""               # dir to save classification heads
    config_name = '8vision_train'               # 8 Vision dataset config name
    
    model = CLIPModel.from_pretrained(vit_path, cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(vit_path, cache_dir=cache_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    # Add preprocessors
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = processor.image_processor
        dataset_config['eval_preprocess'] = processor.image_processor
    config = prepare_experiment_config(raw_config)
    
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    
    os.makedirs(classification_heads_dir, exist_ok=True)
    
    language_encoder = model.text_model.eval().to(device)
    for dataset_name, loader_dict in tqdm(zip(dataset_names, config['data'])):
        print(f'On {dataset_name}')
        template = get_templates(dataset_name)
        clip_encodings = build_classification_head(
            model.eval().to(device), processor.tokenizer, loader_dict['test']['class_names'], template, device=device
        )
        torch.save(clip_encodings, os.path.join(classification_heads_dir, f'{dataset_name}_head.pt'))