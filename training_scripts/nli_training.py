import os
# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from huggingface_hub import login
# Get the token from environment variables
token = os.getenv('HUGGINGFACE_TOKEN')
login(token=token)

import torch
from  torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from peft import (
    get_peft_model,
    LoraConfig,
)
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import evaluate_logits

import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def grab_nli_loader_fns(name):
    """ Returns the dataset loader functions for the specified NLI dataset """
    if name == 'snli':
        from dataset.snli import prepare_train_loaders, prepare_test_loaders
    elif name == 'mnli':
        from dataset.mnli import prepare_train_loaders, prepare_test_loaders
    elif name == 'sick':
        from dataset.sick import prepare_train_loaders, prepare_test_loaders
    elif name == 'qnli':
        from dataset.qnli import prepare_train_loaders, prepare_test_loaders
    elif name == 'rte':
        from dataset.rte import prepare_train_loaders, prepare_test_loaders
    elif name == 'scitail':
        from dataset.scitail import prepare_train_loaders, prepare_test_loaders
    else:
        raise NotImplementedError(name)
    
    return prepare_train_loaders, prepare_test_loaders


def grab_nli_dataset_configs(name):
    """ Returns the dataset config for the specified NLI dataset """
    if name == 'snli':
        from dataset.configs import snli as base_config
    elif name == 'mnli':
        from dataset.configs import mnli as base_config
    elif name == 'sick':
        from dataset.configs import sick as base_config
    elif name == 'qnli':
        from dataset.configs import qnli as base_config
    elif name == 'rte':
        from dataset.configs import rte as base_config
    elif name == 'scitail':
        from dataset.configs import scitail as base_config
    else:
        raise NotImplementedError(name)
    return base_config

#----------------- Edit from here -----------------#
PTM_MODEL_PATH = ""                                                     # Path to the pre-trained model checkpoint
CACHE_DIR = ""                                                          # Path to the cache directory
MODEL_SAVE_DIR = ""                                                     # Directory to save the model
MAX_NUM_EPOCHS = 0                                                      # Maximum number of epochs
MAX_STEPS = 0                                                           # Max steps for training
EVAL_AFTER_STEPS = 4000                                                 # Evaluate model after these many steps
TASK = 'qnli'                                                           # Task to train model on
PREPARE_TRAIN_LOADERS, PREPARE_TEST_LOADERS = grab_nli_loader_fns(TASK) # Grab the dataset loader functions for the specified NLI dataset
DATASET_CONFIG = grab_nli_dataset_configs(TASK)                         # Grab the dataset config for the specified NLI dataset
LR = 3e-5                                                               # Learning rate
BATCH_SIZE = 1                                                          # Batch size
NUM_WORKERS = 1                                                         # Number of workers

DATASET_CONFIG['batch_size'] = BATCH_SIZE
DATASET_CONFIG['num_workers'] = NUM_WORKERS
MODEL_NAME_OR_PATH = "meta-llama/Meta-Llama-3-8B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#__________________________________________________#

"""
Original Label info:
snli: 0 - entailment, 1 - neutral, 2 - contradiction
mnli: 0 - entailment, 1 - neutral, 2 - contradiction
sick: 0 - entailment, 1 - neutral, 2 - contradiction
qnli: 0 - entailment, 1 - non-entailment
rte : 0 - entailment, 1 - not-entailment
scitail : entails and neutral 
"""

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"])

cache_dir=CACHE_DIR

train_dataloader = PREPARE_TRAIN_LOADERS(DATASET_CONFIG)['full']
val_dataloader = PREPARE_TEST_LOADERS(DATASET_CONFIG)['val']
test_dataloader = PREPARE_TEST_LOADERS(DATASET_CONFIG)['test']

model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, cache_dir=cache_dir, num_labels = 3)
model = get_peft_model(model,peft_config)

mask_class = DATASET_CONFIG['mask_class']
print(mask_class)

model.load_state_dict(torch.load(PTM_MODEL_PATH, map_location=torch.device(DEVICE)))
print(model.print_trainable_parameters())

tokenizer = PREPARE_TRAIN_LOADERS(DATASET_CONFIG)['tokenizer']
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

optimizer = AdamW(params=model.parameters(), lr=LR)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * MAX_NUM_EPOCHS),
    num_training_steps=(len(train_dataloader) * MAX_NUM_EPOCHS),
)

criterion = CrossEntropyLoss()
model = model.to(DEVICE)

print(f"LoRA Task is: {TASK}")
total_steps = 0
for epoch in range(MAX_NUM_EPOCHS):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        total_steps += 1
        batch.to(DEVICE)
        outputs = model(**batch)
        if mask_class != None:
            outputs.logits[:, mask_class] = -1e10
        loss = criterion(outputs.logits, batch.labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if total_steps%EVAL_AFTER_STEPS == 0:
            model.eval()
            acc = evaluate_logits(model, val_dataloader, DEVICE, mask_class = mask_class)
            print(f"epoch {epoch} val acc :", acc)
            model_save_path = os.path.join(MODEL_SAVE_DIR, TASK+"weighted_loss"+str(total_steps)+'.pt')
            torch.save(model.state_dict(), model_save_path)
        if total_steps >= MAX_STEPS:
            break
    model.eval()
    
acc = evaluate_logits(model, test_dataloader, DEVICE, mask_class = mask_class)
print(f"test acc :", acc)