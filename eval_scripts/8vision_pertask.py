import os
import pdb
import torch
from copy import deepcopy

import numpy as np
from utils import *
from task_merger import get_merge_handler

def run_BIG_function():
    EVAL_SPLIT = 'test'
    BIGSEED = 420
    set_seed(BIGSEED)
    # Get config
    CONFIG_NAME = 'vitB_r16_knots_ties'
    # CONFIG_NAME = 'vitL_r16_knots_ties'
    # CONFIG_NAME = 'vitL_r16_knots_dare_ties'
    # CONFIG_NAME = 'vit_b_r16_knots_dare_ties'
    print("Running with config: ", CONFIG_NAME)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    
    fine_tuned_acc = {
        'stanford_cars' : 74.0,
        'dtd' : 58.3,
        'eurosat' : 99.0,
        'gtsrb' : 92.7,
        'mnist' : 99.3,
        'resisc45' : 88.4,
        'sun397' : 64.5,
        'svhn' : 96.2
    }
    
    # fine_tuned_acc_rank16_ViTL14 = {
    #     'stanford_cars' :99.76682729675113,
    #     'dtd' : 70.0531914893617,
    #     'eurosat' : 98.59259259259259,
    #     'gtsrb' : 97.19912905779889,
    #     'mnist' : 99.525,
    #     'resisc45' : 95.69841269841269,
    #     'sun397' : 79.59697732997482,
    #     'svhn' : 97.72399884759435,
    # }
    # fine_tuned_acc = fine_tuned_acc_rank16_ViTL14
    
    print(raw_config['task_merge_config'])
    with torch.no_grad():
        all_results = deepcopy(config['task_merge_config'])
        print('Creating Merge')
        # iniitalize merging function
        models = np.array([i.cpu() for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
                deepcopy(models), 
                pretrained_model=deepcopy(config['models']['new']), 
                param_handler=config['param_handler'],
                device=device,
                merge_config=config['task_merge_config'],
            )
        Merge.transform(config['task_merge_config'])
        # set task scaling coefficients
        Merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])
        merged_model = Merge.merge(config['task_merge_config'])
        print('Evaluate Merged Model on Each Dataset')
        print("Using config: ", config['task_merge_config'])
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i]+'_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] *100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100)/ fine_tuned_acc[dataset_names[i]] *100
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        print('Finished!')
    
if __name__ == "__main__":
    run_BIG_function()
        