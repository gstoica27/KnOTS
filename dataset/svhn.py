import os
import torch
from torchvision.datasets import SVHN as PyTorchSVHN


ROOT = "" # Path to the root directory of the dataset

class SVHN:
    def __init__(self,
                 is_train,
                 preprocess,
                 location=ROOT,
                 batch_size=128,
                 num_workers=16):

        # to fit with repo conventions for location
        modified_location = os.path.join(location, 'svhn')

        if is_train:
            self.train_dataset = PyTorchSVHN(
                root=modified_location,
                download=True,
                split='train',
                transform=preprocess
            )

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

        else:
            self.test_dataset = PyTorchSVHN(
                root=modified_location,
                download=True,
                split='test',
                transform=preprocess
            )

            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
def prepare_train_loaders(config):
    dataset_class = SVHN(
        is_train=True,
        preprocess=config['train_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders

def prepare_test_loaders(config):
    dataset_class = SVHN(
        is_train=False,
        preprocess=config['eval_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {
        'test': dataset_class.test_loader
    }
    if config.get('val_fraction', 0) > 0.:
        print('splitting svhn')
        test_set = loaders['test'].dataset
        # test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        # shuffled_idxs = np.random.permutation(np.arange(len(test_set)))
        shuffled_idxs = torch.load(config['shuffled_idxs'])
        num_valid = int(len(test_set) * config['val_fraction'])
        valid_idxs, test_idxs = shuffled_idxs[:num_valid], shuffled_idxs[num_valid:]
        
        # test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        val_set =  torch.utils.data.Subset(test_set, valid_idxs)
        test_set =  torch.utils.data.Subset(test_set, test_idxs)
        loaders['test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['val'] = torch.utils.data.DataLoader(
            val_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    loaders['class_names'] = dataset_class.classnames
    
    return loaders
