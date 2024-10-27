import os
import pdb
import torch
from copy import deepcopy

import numpy as np
from utils import *
from task_merger import get_merge_handler

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="1"

import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

from huggingface_hub import login
# Get the token from environment variables
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)

def run_BIG_function():
    CONFIG_NAME = 'llama8B_r16_knots_ties'
    TASK_HEADS_PATH = "/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ingredients/all6_masked_zeroed_heads.pt"
    COMPUTE_TRANSFORM = False
    EVAL_SPLIT = 'val'
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    print(raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")
    
    transform_listified = [str(i) if k != 'ingredients_path' else os.path.basename(i).replace('.pt', '') for k, i in raw_config['task_merge_config'].items()]
    transform_listified += [str(v) for k, v in raw_config['model']['ft_config'].items() if k in {'r', 'type', 'lora_alpha'}]
    csv_file = os.path.join(
        './csvs_new',
        ":".join(dataset_names),
        raw_config['model']['name'],
        raw_config['eval_type'],
        ":".join(transform_listified),
        f'{EVAL_SPLIT}.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f'Saving results to {csv_file}')

    # Parameters are tuned in the order specified in search_config
    default_params = {
        'scaling_coeffs': 1.0,
        'topK' : 40,
        # 'dare_pruning_coeffs':0.9
    }  #Default config
    
    order_of_processing_params = [ 
        'scaling_coeffs',
        'topK',
        # 'dare_pruning_coeffs'
        ]
    
    task_heads = torch.load(TASK_HEADS_PATH)
    search_config = {
        # 'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        # 'scaling_coeffs': np.arange(0.1, 1.1, step=0.1),
        # 'topK': (np.arange(2, 5, step=1) * 10),
        'scaling_coeffs': [1.0],
        'topK': [40],
    }
    print(f"default params: {default_params}")
    print(f"order_of_processing_params: {order_of_processing_params}")

    finetuned_llama3_acc = {
        'snli': 92.49796416938111,
        'mnli': 90.30820173204279,
        'sick': 91.58173664900122,
        'qnli' : 94.48512585812358,
        'rte' : 89.85507246376812,
        'scitail': 96.51928504233303,
    }
    
    print("Using Llama fine-tuned acc")
    fine_tuned_acc = finetuned_llama3_acc
    
    
    print(f'Finetuned Accs: {fine_tuned_acc}')
    print(search_config)
    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        merged_model = Merge.merge(config['task_merge_config'])

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1

        print('Evaluate Merged Model on Each Dataset')
        device = 'cuda'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad(): 
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i])
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
        Merge = MergeClass(
                models,
                pretrained_model=config['models']['new'], 
                param_handler=config['param_handler'],
                device=device,
                merge_config=config['task_merge_config'],
            )
        
        if COMPUTE_TRANSFORM:
            Merge.transform(config['task_merge_config'])
            
        print(config['task_merge_config'])
        for param in order_of_processing_params:
            best_val_results = {'Average_norm_acc' : 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] =  value
                all_results = merge_and_eval(Merge, EVAL_SPLIT = EVAL_SPLIT, instance_params = instance_params)
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
            test_result = merge_and_eval(Merge, EVAL_SPLIT = 'test', instance_params =instance_params)
            print(test_result)
            
if __name__ == "__main__":
    run_BIG_function()
        