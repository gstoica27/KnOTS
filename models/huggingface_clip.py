import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
from peft import get_peft_model, LoraConfig
from collections import defaultdict


CACHE_DIR = ""  # Set this to the directory where you want to cache the models
MODEL_NAME = "" # Set this to the model name you want to use

class HFCLIPVisionModel(nn.Module):
    def __init__(
        self, model_name=MODEL_NAME, 
        cache_dir=CACHE_DIR, 
        device='cpu'
    ):
        super().__init__()
        self.device = device
        model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.vision_head = model.visual_projection.to(device)
        self.vision_model = deepcopy(model.vision_model)
        
        self.train_preprocess = lambda x: processor.image_processor(x, return_tensors='pt')
        self.val_preprocess = lambda x: processor.image_processor(x, return_tensors='pt')
        
        # THIS IS VERY IMPORTANT TO PASS THROUGH FOR HF ADAPTERS
        self.config = model.config
        self.vision_head.weight.requires_grad = False
    
    def forward(self, x):
        # If we have a buggy return from processors, fix it
        if len(x['pixel_values'].shape) == 5:
            x['pixel_values'] = x['pixel_values'].squeeze(1)
            
        return self.vision_head(self.vision_model(**x)[1])
    
    def get_base_model(self):
        return self


class HFLoRACLIPVisionModel(nn.Module):
    def __init__(
        self, model_name=MODEL_NAME, 
        cache_dir=CACHE_DIR, 
        lora_config=None, device='cpu'
    ):
        super().__init__()
        self.device = device
        model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        valid_args = list(LoraConfig.__dict__.keys())
        lora_config = LoraConfig(**{k: v for k,v in lora_config.items() if k in valid_args})
        model.vision_model = get_peft_model(model.vision_model, lora_config).to(device)
        self.vision_model = deepcopy(model.vision_model)
        self.vision_head = model.visual_projection.to(device)
        self.vision_head.weight.requires_grad = False
        # Set Processing
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.train_preprocess = lambda x: processor.image_processor(x, return_tensors='pt')
        self.val_preprocess = lambda x: processor.image_processor(x, return_tensors='pt')
        # Run model without adapters
        self.disable_adapters = False
    
    def forward(self, x):
        # If we have a buggy return from processors, fix it
        if isinstance(x, torch.Tensor):
            x = {'pixel_values': x}
        if len(x['pixel_values'].shape) == 5:
            x['pixel_values'] = x['pixel_values'].squeeze(1)
        
        if self.disable_adapters:
            with self.vision_model.disable_adapter():
                vision_encodings = self.vision_model(**x)
        else:
            vision_encodings = self.vision_model(**x)
        text_encoding = self.vision_head(vision_encodings[1])
        return text_encoding
    
    def replace_sd_keys(self, sd, original, new):
        new_sd = {}
        for key, val in sd.items():
            new_key = key.replace(original, new)
            new_sd[new_key] =  val
        return new_sd
    
    def get_base_model(self):
        self.model.vision_model = self.model.vision_model.get_base_model()
        return self.model
    
def get_model_from_config(config, device):
    if config.get('ft_config', defaultdict(lambda: None))['type'] == 'lora':
        model = HFLoRACLIPVisionModel(
            model_name=config['base_type'],
        cache_dir=config['cachedir'],
        lora_config=config['ft_config'],
        device=device
        )
    else:
        model = HFCLIPVisionModel(
            model_name=config['base_type'],
            cache_dir=config['cachedir'],
            device=device,
        )
    return model

