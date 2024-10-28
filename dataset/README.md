# Datasets
This file provides explanations for understanding the structure of the datasets directory.

## Adding Datasets
Adding a new dataset requires three (optionally four) parts. 

#### Dataset Definition
First, please define your new dataset following the same pattern that exists in each currently existing dataset file. 
This basically entails writing two functions "prepare_train_loaders" and "prepare_test_loaders". 
Each of these functions prepares (1) a dataloader for the full dataset, and (2) optionally a number dataloaders corresponding to subdatasets of the full dataset. 
These correspond to dataloaders that only use a subset of the full data images and labels (e.g., any dataset that does not have a validation set by default). 

#### Add Dataset Config
Second, you will need to add a config to describe the dataset you've created. 
This can be done by adding a succinct dictionary to the "configs.py" file in this directory. 
Please see the file for more information. Below is an example config:
```python
stanford_cars = {
    'wrapper': Cars,
    'batch_size': 128,
    'res': 224,
    'type': 'stanford_cars',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stanford_cars'
}
```
Here the variable name *corresponds to the name of the dataset provided in the experiment config*

#### (Optional) Extract CLIP Encodings
If your experiment involves a CLIP-style model, you need to extract the CLIP label encodings for your task. 
You can do this by running "parsing/generate_clip_heads.py" and following the instructions there.
Depending on the dataset you wish to train, you may need to also add a set of label templates pertaining to the dataset in the "templates.py" file.

#### Add Dataset Parser
The last step entails adding a couple lines to parse your newly added dataset, in the utils.py file. 
Line 368 (prepare_data) contains the information you want. 
All that needs to be done is adding an elif that checks for the name of your dataset, and if it is a match, loads in the train and test loaders you've just created, and passes them through the pipeline.

That's it! You now have added a new dataset. 
