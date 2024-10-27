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
        'test': dataset_class.test_loader,
        'val' : dataset_class.val_loader
    }
    return loaders
        





