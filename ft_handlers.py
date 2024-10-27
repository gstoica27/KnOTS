import torch.nn as nn
from collections import defaultdict, OrderedDict

"""
True base_model.model.classifier.original_module.dense.weight
True base_model.model.classifier.original_module.dense.bias
True base_model.model.classifier.original_module.out_proj.weight
True base_model.model.classifier.original_module.out_proj.bias

"""


class LoRAHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.base_model.state_dict()
        for key, val in sd.items():
            if 'lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif 'lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
        
        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
            task_parameters[name] = (key2val['B'] @ key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))
    
    def get_model(self):
        return self.base_model.get_base_model
    
    
class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model