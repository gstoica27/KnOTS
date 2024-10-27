from torch.utils.data import DataLoader
from .huggingface_datasets import HuggingFaceDataset


def prepare_train_loaders(config):
    dataset_class = HuggingFaceDataset(
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
    dataset_class = HuggingFaceDataset(
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
        val_test = dataset['validation'].train_test_split(test_size = config['val_fraction'], shuffle=False, seed = 42)
        test_loader = DataLoader(val_test["train"], shuffle=False, collate_fn=dataset_class.collate_fn, batch_size=dataset_class.batch_size, num_workers = dataset_class.num_workers)
        val_loader = DataLoader(val_test["test"], shuffle=False, collate_fn=dataset_class.collate_fn, batch_size=dataset_class.batch_size, num_workers = dataset_class.num_workers)
        loaders['test'] = test_loader
        loaders['val'] = val_loader
    return loaders
        





