from collections import defaultdict, OrderedDict
from copy import deepcopy
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_mask_fn


class LinearCombiner(nn.Module):
    def __init__(self, base_linear, add_weight, device=0):
        super().__init__()
        self.base_linear = base_linear
        self.add_weight = add_weight
        self.device = device
        
        self.weight = self.base_linear.weight + self.add_weight
        bias = None
        if hasattr(self.base_linear, 'bias'):
            bias = self.base_linear.bias
        self.bias = bias
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    

class SingularValsTrainer(nn.Module):
    def __init__(self, ingredients, merge_config, singular_weights, base_model, device='cpu'):
        super(SingularValsTrainer, self).__init__()
        self.device = device
        self.ingredients = ingredients
        self.merge_config = merge_config
        
        singular_weights = []
        self.parameter_list = nn.ParameterList()
        self.list_of_key2pm_idx = []
        
        for idx, Ss in enumerate(ingredients['task_Ss']):
            task_weights = {}
            key2pm_idx = {}
            for jdx, (key, val) in enumerate(Ss.items()):
                task_weights[key] = nn.Parameter(deepcopy(val) * .2, requires_grad=True).to(device)
                self.parameter_list.append(task_weights[key])
                key2pm_idx[key] = idx * len(Ss) + jdx 
            singular_weights.append(task_weights)
            self.list_of_key2pm_idx.append(key2pm_idx)
            
            
        self.singular_weights = singular_weights
        self.base_model = base_model
        self.create_merged_model()
    
    def forward(self, x):
        return self.model(x)
    
    def create_merged_model(self):
        list_to_device = lambda x: [{key: val.to(self.device) for key, val in elem.items()} for elem in x]
        dict_to_device = lambda x: {key: val.to(self.device) for key, val in x.items()}
        
        ingredients = deepcopy(self.ingredients)
        ftms_others = self.make_trainable(list_to_device(ingredients['ftms_others']))
        ptm_reference_params = self.make_trainable(dict_to_device(ingredients['ptm_reference_params']))
        U = self.make_trainable(dict_to_device(ingredients['U']))
        task_sVs = self.make_trainable(list_to_device(ingredients['task_sVs']))
        task_vnorms = self.make_trainable(list_to_device(ingredients['task_vnorms']))
        pre_mask_fns = ingredients['pre_mask_fns']
        pre_merge_fns = ingredients['pre_merge_fns']
        representations = self.directions_to_reps(task_sVs)
        ftms_reps, ptm_rep = self.apply_pre_mask_fns(pre_mask_fns, representations)
        mask_fn = get_mask_fn(self.merge_config['mask_method'])
        masks = mask_fn(ftms_reps, **self.merge_config)
        ftms_reps = torch.vstack(ftms_reps).clone()
        masked_sVs = ftms_reps * masks
        pre_merge_sVs = self.apply_pre_merge_fns(pre_merge_fns, masked_sVs, masks, ptm_rep)
        pre_merge_sVs_dict = self.rep_to_state_dict(pre_merge_sVs, task_sVs[0])
        # ------------------------------------- We start here -------------------------------------
        pre_merge_sVs = self.apply_Ss_on_Vs(pre_merge_sVs_dict)
        rescaled_Vs = self.rescale_Vs(pre_merge_sVs, task_vnorms)
        template_sd = {key: val.detach() for key, val in rescaled_Vs[0].items()}
        mask_sd = self.mask_to_state_dict([m.cuda() for m in masks], template_sd)
        merged_sV_sd = self.weighted_merge(
            merging_type=self.merge_config.get('merging_type', 'mean'), 
            task_Vs=rescaled_Vs,
            task_masks=mask_sd
        )
        merged_sd = self.reconstruct_merged_sd(U, merged_sV_sd)
        merged_others = self.merge_others(ftms_others)
        merged_sd = self.add_others(merged_sd, merged_others)
        # pdb.set_trace()
        merged_sd = self.matrix_to_state_dict(merged_sd, ptm_reference_params)
        # Add merged sd to the ptm
        merged_base = deepcopy(self.base_model).to(self.device).train()
        # for parameter in merged_base.parameters():
        #     parameter.requires_grad = True
        merged_model = self.add_trainable_parameters(
            merged_base, merged_sd,  
            concat_across_output=self.merge_config.get('concat_across_output', True)
        )
        # pdb.set_trace()
        self.model = merged_model
        # return merged_model
    
    def replace_Linear_with_LinearCombiner(self, model, key, add_weight):
        stages = key.split('.')
        x = getattr(model, stages[0])
        for stage in stages[1:-1]:
            if stage in [str(i) for i in range(20)]:
                x = x[int(stage)]
                continue
            x = getattr(x, stage)
        # pdb.set_trace()
        module = LinearCombiner(getattr(x, stages[-1]), add_weight)
        setattr(x, stages[-1], module)
    
    def add_trainable_parameters(self, base_model, parameters, concat_across_output = True, scaling_coeffs=1.):
        sd = dict(base_model.named_parameters())
        for key, val in parameters.items():
            cur_val = deepcopy(sd[key])
            try:
                if (concat_across_output):
                    # sd[key] = sd[key] + val * scaling_coeffs
                    # sd[key].add_(val * scaling_coeffs)
                    self.replace_Linear_with_LinearCombiner(base_model, key.replace('.weight', ''), val * scaling_coeffs)
                else:
                    # sd[key] = sd[key] + val.T * scaling_coeffs
                    sd[key].add_(val.T * scaling_coeffs)
            except:
                pdb.set_trace()
        # pdb.set_trace()
        # base_model.load_state_dict(sd)
        return base_model
    
    def make_trainable(self, d):
        return d
        # if isinstance(d, list):
        #     return [self.make_trainable(elem) for elem in d]
        # for key, val in d.items():
        #     d[key].requires_grad = True
        # return d
    
    def get_layer_names(self, state_dict):
        layer_names = defaultdict(lambda: dict())
        for key in state_dict:
            if ('.weight' in key) or ('_weight' in key):
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif ('.bias' in key) or ('_bias' in key):
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'
        return layer_names
    
    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]
        
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
                
        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        # pdb.set_trace()
        for layer_name, value in matrix.items():
            try:
                parameter_types = layer_names[layer_name.replace(':other', '')]
                if 'other' in parameter_types:
                    # pdb.set_trace()
                    name = parameter_types['other'].replace(':other', '')
                    merged_state_dict[name] = value.reshape(reference_dict[name].shape)
                else:
                    # weight_name = parameter_types['weight']
                    if 'bias' in parameter_types: 
                        bias_index = value.shape[1] - 1
                        value, bias = value[:, :bias_index], value[:, -1].flatten()
                        merged_state_dict[parameter_types['bias']] = bias
                    if 'norm' in layer_name or 'ln' in layer_name:
                        value = torch.diagonal(value)
                    name = parameter_types['weight']
                    merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))
            except:
                pdb.set_trace()
                    
        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict
    
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )
        
    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        # pdb.set_trace()
        if isinstance(vector, list) or len(vector.shape) == 2:
            # pdb.set_trace()
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict
    
    def apply_pre_mask_fns(self, fns, sds, ptm_sd=None):
        for fn in fns:
            sds, ptm_sd = fn(sds, ptm_sd)
        return sds, ptm_sd
    
    def apply_pre_merge_fns(self, fns, ftms, masks, ptm=None):
        for fn in fns:
            ftms, masks, ptm = fn(ftms, masks, ptm)
        return ftms
    
    def apply_Ss_on_Vs(self, task_Vs):
        task_sVs = [dict() for i in range(len(task_Vs))]
        for idx, (Vs, key2pm_idx) in enumerate(zip(task_Vs, self.list_of_key2pm_idx)):
            for key, V in Vs.items():
                pm_idx = key2pm_idx[key]
                s = F.relu(self.parameter_list[pm_idx])
                task_sVs[idx][key] = torch.diag(s) @ V
        return task_sVs
    
    def rescale_Vs(self, task_Vs, task_vnorms):
        # pdb.set_trace()
        taskv_rescaled = [dict() for _ in range(len(task_Vs))]
        for idx, (task_v, task_vnorm) in enumerate(zip(task_Vs, task_vnorms)):
            for key in task_v.keys():
                if task_vnorm[key] is not None:
                    taskv_rescaled[idx][key] = task_v[key] * task_vnorm[key]
                else:
                    taskv_rescaled[idx][key] = task_v[key]
        return taskv_rescaled
    
    def reconstruct_merged_sd(self, U_sd, sV_sd):
        if isinstance(sV_sd, list):
            if isinstance(U_sd, list):
                return [self.reconstruct_merged_sd(U, sV) for U, sV in zip(U_sd, sV_sd)]
            return [self.reconstruct_merged_sd(U_sd, sV) for sV in sV_sd]
        sd = {}
        for key, U in U_sd.items():
            sd[key] = (U @ sV_sd[key]).to(torch.float32) # ensure float32 dtype
        return sd
    
    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]
        
        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats
    
    def merge_others(self, ftms_others, weights=None):
        merged_others = {}
        for key in ftms_others[0].keys():
            pdb.set_trace
            if weights is not None:
                merged_others[key] = torch.stack([ftm_other[key] * weight.flatten() for ftm_other, weight in zip(ftms_others, weights)], dim=0).sum(dim=0)
            else:
                merged_others[key] = torch.stack([ftm_other[key] for ftm_other in ftms_others], dim=0).sum(dim=0)
        return merged_others
    
    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)
    
    def weighted_merge(self, merging_type, task_Vs, task_masks, weights=None):
        merged_Vs = {}
        for key in task_Vs[0].keys():
            # pdb.set_trace()
            stacked_Vs = torch.stack([task_V[key] for task_V in task_Vs], dim=0)
            stacked_mask = torch.stack([task_mask[key] for task_mask in task_masks], dim=0)
            
            if weights is not None:
                stacked_Vs = stacked_Vs * weights[:, None, None]
            
            if merging_type == "mean":
                # pdb.set_trace()
                non_zero_counts = (stacked_mask != 0).sum(dim=0).float()
                denominator = non_zero_counts.clamp(min=1)
                merged_Vs[key] = (stacked_Vs).sum(dim=0) / denominator
            elif merging_type == "sum":
                merged_Vs[key] = stacked_Vs.sum(dim=0)
            elif merging_type == 'max':
                merged_Vs[key] = stacked_Vs.max(dim=0).values
            else:
                raise ValueError(f'Unknown merging type: {merging_type}. Pick from mean, sum, or max')
        return merged_Vs