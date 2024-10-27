import os
import torch
from copy import deepcopy
import pickle

import numpy as np
from utils import *
from task_merger import get_merge_handler

def run_BIG_function():
    EVAL_SPLIT = 'val'
    AUX_INFO = 'topK'
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    # Get config
    CONFIG_NAME = 'vitB_r16_knots_ties'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    # Get clip encodings
    joint_stuff_dir = "./dataset/8vision_joint_components"
    print(f'Loading joint encodings from {joint_stuff_dir}')
    joint_encodings = get_clip_encodings(os.path.join(joint_stuff_dir, 'joint_head.pt'))
    dataset_mappers = pickle.load(open(os.path.join(joint_stuff_dir, 'joint_mappers.pkl'), 'rb'))
    # joint_class_names = pickle.load(open(os.path.join(joint_stuff_dir, 'joint_class_names.pkl'), 'rb'))
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
    default_params = {'scaling_coeffs': .6, 'topK' : 30, 'pre_svd_topK' : 100, 'interference_threshold': 0.0} #Default config
    order_of_processing_params = [
        'scaling_coeffs', 
        # 'interference_threshold', 
        'topK'
    ]
    search_config = {
        # 'scaling_coeffs': np.arange(0.1, 2.1, step=0.1),
        # 'topK': (np.arange(1, 11, step=1) * 10)[::-1],
        'scaling_coeffs': [.6],
        'topK': [30],
    }
    
    print(search_config)
    param_names, values = zip(*search_config.items())
    def merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params = None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        print('Creating Merge')
        # set task scaling coefficients
        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        merged_model = Merge.merge(config['task_merge_config'])
        
        print('Evaluate Merged Model on Each Dataset')
        joint_topk = defaultdict(lambda: 0)
        joint_total = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            
            label_map = torch.from_numpy(dataset_mappers[dataset_names[i]]['local2joint_map']).to(device)
            topk_counts, total, topk, confusion_matrix = evaluate_cliphead_joint(
                merged_model.to(device), 
                loader, 
                class_vectors=joint_encodings.to(device), 
                aux_class_map=label_map, 
            )
            
            joint_total += total
            for k, count in topk_counts.items():
                joint_topk[k] += count
            print(f"TopK for {dataset_names[i]}:")

            topk_prepared = {f"Top-{k}": f"{np.round(v * 100, 3)}" for k, v in topk.items()}
            print("\t".join([f"{k}: {v}" for k, v in topk_prepared.items()]))
            for k, v in topk_prepared.items():
                all_results[dataset_names[i] + f" {k}"] = v
            
            
        for k, v in joint_topk.items():
            all_results[f'Joint Top-{k}'] = np.round(v / joint_total * 100, 3)
        print(f"Joint TopK:")
        topk_prepared = {f"Top-{k}": f"{np.round(v / joint_total * 100, 3)}" for k, v in joint_topk.items()}
        print("\t".join([f"{k}: {v}" for k, v in topk_prepared.items()]))
        all_results.update(config['task_merge_config'])
        write_to_csv(all_results, csv_file)
        return all_results
        


    with torch.no_grad():
    
        print(search_config)
        param_names, values = zip(*search_config.items())

        models = np.array([i.cpu().eval() for i in config['models']['bases']])
        
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
                deepcopy(models), 
                pretrained_model=deepcopy(config['models']['new']), 
                param_handler=config['param_handler'],
                device=device,
                merge_config=config['task_merge_config'],
            )
        
        Merge.transform(config['task_merge_config'])
        print(config['task_merge_config'])
        # For linear search
        for param in order_of_processing_params:
            best_val_results = {'Joint Top-3' : 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] =  value
                # pdb.set_trace()
                all_results = merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params = instance_params)
                if (all_results['Joint Top-3'] > best_val_results['Joint Top-3']):
                    best_val_results = deepcopy(all_results)
                else:
                    break
            default_params[param] = best_val_results[param]
        # For grid search
        # best_val_results = {'Joint Top-3' : 0.0}
        # for bundle in product(*values):
        #     # pdb.set_trace()
        #     instance_params = dict(zip(param_names, bundle))
        #     all_results = merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params =instance_params)
        #     if (all_results['Joint Top-3'] > best_val_results['Joint Top-3']):
        #         best_val_results = deepcopy(all_results)

        if (EVAL_TEST == True):
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", best_val_results)
            for key in search_config.keys():
                instance_params.update({key : best_val_results[key]})
            test_result = merge_and_eval(Merge, EVAL_SPLIT = 'test', instance_params =instance_params)
            print(test_result)
        print(f'Saved results to {csv_file}')
            
if __name__ == "__main__":
    run_BIG_function()
