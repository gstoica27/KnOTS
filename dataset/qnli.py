from torch.utils.data import DataLoader
from .huggingface_datasets import HuggingFaceDataset


class QNLI(HuggingFaceDataset):
    def __init__(
        self,
        location = None,
        task = None,
        model_name_or_path = None,
        batch_size=32,
        num_workers=16
    ):
        
        super().__init__(location, task, model_name_or_path, batch_size, num_workers)
        
        self.tokenized_datasets = self.tokenized_datasets.map(lambda example: {"labels": 2 if example["labels"] == 1 else example["labels"]}) # 0 is entailment, 2 is non-entailment
        self.train_loader = DataLoader(self.tokenized_datasets["train"], shuffle=True, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)
        self.val_loader = DataLoader(self.tokenized_datasets["validation"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)
        self.test_loader = DataLoader(self.tokenized_datasets["test"], shuffle=False, collate_fn=self.collate_fn, batch_size=batch_size, num_workers = num_workers)


def prepare_train_loaders(config):
    dataset_class = QNLI(
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
    dataset_class = QNLI(
        location=config['dir'],
        task = config["type"],
        model_name_or_path = config["model_name_or_path"],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    loaders = {
        'test': dataset_class.val_loader,
    }
    if config.get('val_fraction', 0) > 0.:
        dataset = dataset_class.tokenized_datasets
        val_test = dataset['validation'].train_test_split(test_size = config['val_fraction'], shuffle=True, seed = 42)
        test_loader = DataLoader(val_test["train"], shuffle=False, collate_fn=dataset_class.collate_fn, batch_size=dataset_class.batch_size, num_workers = dataset_class.num_workers)
        val_loader = DataLoader(val_test["test"], shuffle=False, collate_fn=dataset_class.collate_fn, batch_size=dataset_class.batch_size, num_workers = dataset_class.num_workers)
        loaders['test'] = test_loader
        loaders['val'] = val_loader
    return loaders
        





