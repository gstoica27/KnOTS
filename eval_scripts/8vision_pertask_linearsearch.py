import os
import pdb
import torch
from copy import deepcopy
import numpy as np
from utils import *
from task_merger import get_merge_handler

def run_BIG_function():
    EVAL_SPLIT = 'val'
    AUX_INFO = ''
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    # Get config
    config_name = 'multidataset_hf_clip'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    
    transform_listified = [str(i) for i in list(raw_config['task_merge_config'].values())]
    transform_listified += [str(v) for k, v in raw_config['model']['ft_config'].items() if k in {'r', 'type', 'lora_alpha'}]
    csv_file = os.path.join(
        './csvs',
        ":".join(dataset_names),
        raw_config['model']['name'],
        raw_config['eval_type'],
        ":".join(transform_listified),
        f'{EVAL_SPLIT}_{AUX_INFO}.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f'Saving results to {csv_file}')

    # Parameters are tuned in the order specified in search_config
    default_params = {
        'scaling_coeffs': np.ones(len(dataset_names))*0.1, 
        'topK' : 20, 
        'dare_pruning_coeffs': 1e-5,
        } #Default config
    order_of_processing_params = [
        'scaling_coeffs', 
        # 'topK',
        # 'dare_pruning_coeffs',
    ]
    search_config = {
        'scaling_coeffs': np.arange(0.6, 0.7, step=0.1)[::-1],
        # 'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        # 'topK': (np.arange(1, 11, step=1) * 10),
        }
    
    fine_tuned_acc_rank16_ViTL14 = {
        'stanford_cars' :99.76682729675113,
        'dtd' : 70.0531914893617,
        'eurosat' : 98.59259259259259,
        'gtsrb' : 97.19912905779889,
        'mnist' : 99.525,
        'resisc45' : 95.69841269841269,
        'sun397' : 79.59697732997482,
        'svhn' : 97.72399884759435,
    }

    fine_tuned_acc_rank16 = {
        'stanford_cars' : 74.0,
        'dtd' : 58.3,
        'eurosat' : 99.0,
        'gtsrb' : 92.7,
        'mnist' : 99.3,
        'resisc45' : 88.4,
        'sun397' : 64.5,
        'svhn' : 96.2
    }
    
    fine_tuned_acc_rank128 = {
        'stanford_cars' : 77.3,
        'dtd' : 68.8,
        'eurosat' : 98.5,
        'gtsrb' : 96.9,
        'mnist' : 99.6,
        'resisc45' : 92.2,
        'sun397' : 67.4,
        'svhn' : 97.1
    }
    
    fine_tuned_acc_fft = {
        'stanford_cars' : 77.7,
        'dtd' : 79.4,
        'eurosat' : 99.7,
        'gtsrb' : 98.7,
        'mnist' : 99.7,
        'resisc45' : 96.1,
        'sun397' : 75.3,
        'svhn' : 97.5
    }

    if config['model']['base_type'] == "openai/clip-vit-large-patch14":
        print("Using ViT-L14 rank16 acc to normalize")
        fine_tuned_acc = fine_tuned_acc_rank16_ViTL14
    elif config['model']['ft_config']['r'] == 128 and config['model']['base_type'] == "openai/clip-vit-base-patch32":
        print("Using rank128 acc to normalize")
        fine_tuned_acc = fine_tuned_acc_rank128
    elif config['model']['ft_config']['r'] == 16 and config['model']['base_type'] == "openai/clip-vit-base-patch32":
        print("Using rank16 acc to normalize")
        fine_tuned_acc = fine_tuned_acc_rank16
    elif config['model']['ft_config']['type'] == 'fft' and config['model']['base_type'] == "openai/clip-vit-base-patch32":
        print("Using fft fine-tuned acc")
        fine_tuned_acc = fine_tuned_acc_fft
    
    
    print(f'Finetuned Accs: {fine_tuned_acc}')
    def merge_and_eval(EVAL_SPLIT = 'val', param_handler=None, instance_config = None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        # iniitalize merging function
        print('Creating Merge')
        Merge = MergeClass(
            deepcopy(models), 
            pretrained_model=deepcopy(config['models']['new']), 
            param_handler=param_handler,
            device=device,
            merge_config=instance_config,
        )
        Merge.transform(instance_config)
        # set task scaling coefficients
        Merge.set_scaling_coeffs(instance_config['scaling_coeffs'])
        merged_model = Merge.merge(instance_config)
        
        print('Evaluate Merged Model on Each Dataset')
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
        write_to_csv(all_results, csv_file)
        return all_results
        


    with torch.no_grad():
    
        print(search_config)
        models = np.array([i.cpu().eval() for i in config['models']['bases']])
        
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        print(config['task_merge_config'])
        for param in order_of_processing_params:
            best_val_results = {'Average_norm_acc' : 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] =  value
                config['task_merge_config'].update(instance_params)
                all_results = merge_and_eval(
                    EVAL_SPLIT='val', 
                    param_handler=config['param_handler'], 
                    instance_config=config['task_merge_config']
                )
                if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                    best_val_results = deepcopy(all_results)
                else:
                    break
            default_params[param] = best_val_results[param]

        if (EVAL_TEST == True):
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", best_val_results)
            for key in search_config.keys():
                instance_params.update({key : best_val_results[key]})
            # test_result = merge_and_eval(Merge, EVAL_SPLIT = 'test', instance_params =instance_params)
            config['task_merge_config'].update(instance_params)
            test_result = merge_and_eval(
                EVAL_SPLIT='test', 
                param_handler=config['param_handler'], 
                instance_config=config['task_merge_config']
            )
            print(test_result)
            
if __name__ == "__main__":
    run_BIG_function()
        