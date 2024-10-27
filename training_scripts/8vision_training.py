import os
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from copy import deepcopy
from models.huggingface_clip import HFLoRACLIPVisionModel
import wandb


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0.0
        self.best_test_acc = 0.0
        self.best_train_acc = 0.0

    def early_stop(self, performance_log, model, model_save_path):
        if performance_log['Val Acc'] > (self.max_validation_acc + self.min_delta):
            self.max_validation_acc = performance_log['Val Acc']
            self.best_test_acc = performance_log['Test Acc']
            self.best_train_acc = performance_log['Training Acc']
            save_model(model, model_save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_cliphead_lora(model, train_loader, val_loader, test_loader, class_vectors, remap_class_idxs=None, eval_class_vectors=None, clip_mapper=None, training_config = None, model_save_path = None):
    """Train a cliphead model.
    
    Args:
        model: cliphead model
        train_loader: dataloader to train on
        val_loader: dataloader to validate on
        test_loader: dataloader to test on
        class_vectors: clip label encodings
        remap_class_idxs: array or mapping from true class labels to those expected given the task
        epochs: number of epochs to train for
    Returns:
        model: trained cliphead model
        train_acc: training accuracy
    """
    # wandb.log(training_config)
    epochs = training_config['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr= training_config['lr'], weight_decay=training_config['wd'])
    ne_iters = len(train_loader)
    scheduler = cosine_lr(optimizer, training_config['lr'], training_config['warm_up'], training_config['max_steps'])
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing = training_config['label_smoothing'])
    device = get_device(model)
    
    val_acc = 0.0
    pbar = tqdm(range(epochs), desc=f'finetuning, prev acc: {val_acc}: ')
    early_stopper = EarlyStopper(patience=training_config['early_stopping_patience'], min_delta=training_config['early_stopping_min_delta'])
    end = False
    for epoch in pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            model = model.train()
            step = i + epoch * ne_iters
            optimizer.zero_grad(set_to_none=True)
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            logits = (100.0 * normed_encodings @ class_vectors.T)
    
            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels].to(device)
            else:
                remapped_labels = labels.to(device)
            loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler(step)   

            if (step+1) % training_config['eval_freq'] == 0:
                train_acc, train_loss = evaluate_cliphead(model, train_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss = True)
                val_acc, val_loss = evaluate_cliphead(model, val_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss = True)
                test_acc, test_loss = evaluate_cliphead(model, test_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss = True)
                pbar.set_description(f'finetuning, prev acc: {val_acc}: ')
                print(f'Steps {step}, Test Acc: {test_acc*100.0}, Test Loss: {test_loss}')
                print(f'Steps {step}, Val Acc: {val_acc*100.0}, Val Loss: {val_loss}')
                print(f'Steps {step}, Train Acc: {train_acc*100.0}, Train Loss: {train_loss}')
                performance_log = {"Test Acc" : test_acc, "Test Loss" : test_loss, "Training Loss" : train_loss, "Training Acc" : train_acc, "Val Loss" : val_loss, "Val Acc" : val_acc}
                wandb.log(performance_log, step = step)
                
                if early_stopper.early_stop(performance_log, model, model_save_path): 
                    print(f"Early stopping at : {step}")
                    end = True             
                    break
                print(f'Early Stopping Counter: {early_stopper.counter}')

                if step >= training_config['max_steps'] == 0:
                    end = True
                    break
        if end:
            break
    print("Ending Training")
    print(f" Val Acc @ best ckpt: {early_stopper.max_validation_acc*100.0}, Test Acc @ best ckpt: {early_stopper.best_test_acc*100.0}, Train Acc @ best ckpt: {early_stopper.best_train_acc*100.0}")
    wandb.log({"Val Acc @ best ckpt" : early_stopper.max_validation_acc, "Test Acc @ best ckpt" : early_stopper.best_test_acc, "Train Acc @ best ckpt" : early_stopper.best_train_acc})

    return model, test_acc, test_loss, val_acc, val_loss

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def train_functional(training_config = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)

    if training_config is None:
        training_config = raw_config['training_config']
    print(training_config)

    model_name = VIT_PATH.split('/')[-1]
    wandb.init(project=f"{model_name}_Lora_r{training_config['lora_rank']}_finetuning", config = training_config)

    lc = {
        'r': training_config['lora_rank'],
        'lora_alpha': training_config['lora_rank'],
        'target_modules' :["q_proj", "k_proj", "v_proj", "out_proj"],
        'lora_dropout' :0.1,
        'bias':"none"
        }
    
    lora_config = LoraConfig(
        r=lc['r'],
        lora_alpha=lc['lora_alpha'],
        target_modules=lc['target_modules'],
        lora_dropout=lc['lora_dropout'],
        bias=lc['bias'],
    )

    lora_ptm = HFLoRACLIPVisionModel(
            model_name= VIT_PATH, 
            cache_dir=CACHE_DIR, 
            lora_config=lora_config.__dict__, 
            device=device
    )

    print("Device : ", device)
    print("Config: ", CONFIG_NAME)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = lora_ptm.train_preprocess
        dataset_config['eval_preprocess'] = lora_ptm.val_preprocess
    
    data_loaders = prepare_data(raw_config['dataset'], device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    
    model_save_dir = os.path.join(MODEL_SAVE_DIR, f"lora_rank{lc['r']}")
    
    os.makedirs(model_save_dir, exist_ok=True)
    val_loss = 0
    for dataset_name, loader_dict, class_vectors in tqdm(zip(dataset_names, data_loaders, all_clip_encodings)):
        if dataset_name != training_config['dataset']: continue

        save_path = os.path.join(
            model_save_dir, 
            f"{training_config['dataset']}_lr_{training_config['lr']}_wd_{training_config['wd']}_max_steps_{training_config['max_steps']}_early_stopping_patience_{training_config['early_stopping_patience']}_rescaling_weights{training_config['rescaling_weights']}.pt"
        )
        print(f'Finetuning LoRA on {dataset_name}')
        lora_model = deepcopy(lora_ptm)

        finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_lora(
            lora_model, 
            train_loader = loader_dict['train']['full'],
            val_loader = loader_dict['test']['val'],  
            test_loader = loader_dict['test']['test'],
            class_vectors=class_vectors,
            training_config = training_config,
            model_save_path = save_path
        )
        wandb.finish()
    return val_loss

if __name__ == "__main__":
    # Uncomment ViT name/path for the model you want to train from HF
    # VIT_PATH = "openai/clip-vit-large-patch14"
    VIT_PATH = "openai/clip-vit-base-patch32"   
    CACHE_DIR = ''                                      # Path to cache directory 
    MODEL_SAVE_DIR = ""                                 # Path to save the model
    CONFIG_NAME = '8vision_train'                       # Training config file name
    #Update the training config here or in config file
    training_config={
    # 'epochs' : 10000,
    # 'max_steps' : 100000,
    'early_stopping_patience' : 5,
    'epochs' : 1000,
    'max_steps' : 50000,
    'dataset' : "svhn",
    'eval_freq' : 2000,
    # 'lora_rank' : 4,
    # 'lr' : 8e-4,
    'lora_rank' : 16,
    'lr' : 3e-4,
    # 'lora_rank' : 64,
    # 'lr' : 8e-5,
    # 'lora_rank' : 256,
    # 'lr' : 3e-5,
    'wd' : 1e-1,
    'warm_up' : 500,
    'label_smoothing': 0.0,
    'early_stopping_min_delta' : 1e-3,
    'seed' : 420,
    }
    set_seed(training_config['seed'])

    loss = train_functional(training_config)
    print(loss)
