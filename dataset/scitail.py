from torch.utils.data import DataLoader
import datasets
from  .huggingface_datasets import task_ids, task_to_keys
from transformers import AutoTokenizer


ROOT = "" # Path to the root directory of the dataset

class SCITAIL:
    def __init__(self,
                 location = None,
                 task = None,
                 model_name_or_path = None,
                 batch_size=32,
                 num_workers=16):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        data_format = 'tsv_format'
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        dataset = datasets.load_dataset(task_ids[task], data_format, cache_dir=ROOT)
        sentence1_key, sentence2_key = task_to_keys[task]

        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            outputs = self.tokenizer(*args, truncation=True, max_length=1000)
            return outputs

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns= [t for t in dataset['train'].column_names if t != "label"]
        )
        
        self.tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        self.tokenized_datasets = self.tokenized_datasets.filter(lambda example: example["labels"]=="neutral" or example["labels"]=="entails")
        self.tokenized_datasets = self.tokenized_datasets.map(lambda example: {"labels": 0 if example["labels"] == "entails" else 1 if example["labels"] == "neutral" else example["labels"] })

        if "train" in self.tokenized_datasets:
            self.train_loader = DataLoader(self.tokenized_datasets["train"], shuffle=True, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)
        if "test" in self.tokenized_datasets:
            self.test_loader = DataLoader(self.tokenized_datasets["test"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)
        if "validation" in self.tokenized_datasets:
            self.val_loader = DataLoader(self.tokenized_datasets["validation"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)

    def collate_fn(self, examples):
            return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")
    


def prepare_train_loaders(config):
    dataset_class = SCITAIL(
        location=config['dir'],
        task = config["type"],
        model_name_or_path = config["model_name_or_path"],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader,
        'tokenizer' : dataset_class.tokenizer
    }
    return loaders


def prepare_test_loaders(config):
    dataset_class = SCITAIL(
        location=config['dir'],
        task = config["type"],
        model_name_or_path = config["model_name_or_path"],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    loaders = {
        'test': dataset_class.test_loader,
        'val' : dataset_class.val_loader
    }
    return loaders
        





