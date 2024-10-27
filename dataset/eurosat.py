import os
import torchvision.datasets as datasets
import re
import torch

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


ROOT = "" # Set the root directory to the dataset here


def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out


class EuroSATBase:
    def __init__(self,
                 is_train,
                 train_preprocess,
                 eval_preprocess,
                 location=ROOT,
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        if is_train:
            traindir = os.path.join(location, 'EuroSAT_splits', 'train')

            self.train_dataset = datasets.ImageFolder(traindir, transform=train_preprocess)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            dataset = self.train_dataset
        
        else:
            valdir = os.path.join(location, 'EuroSAT_splits', 'val')
            testdir = os.path.join(location, 'EuroSAT_splits', 'test')
            
            self.val_dataset = datasets.ImageFolder(valdir, transform=eval_preprocess)
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False
            )

            self.test_dataset = datasets.ImageFolder(testdir, transform=eval_preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False
            )
            dataset = self.val_dataset
        
        
        idx_to_class = dict((v, k)
                            for k, v in dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]
    

def prepare_train_loaders(config):
    dataset_class = EuroSATBase(
        is_train=True,
        train_preprocess=config['train_preprocess'], 
        eval_preprocess=config['eval_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders

def prepare_test_loaders(config):
    dataset_class = EuroSATBase(
        is_train=False,
        train_preprocess=config['train_preprocess'], 
        eval_preprocess=config['eval_preprocess'],
        location=ROOT,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {
        'val': dataset_class.val_loader,
        'test': dataset_class.test_loader,
        'class_names': dataset_class.classnames
    }
    
    return loaders

