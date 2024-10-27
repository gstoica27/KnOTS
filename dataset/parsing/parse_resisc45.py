import pdb
import os
import shutil

def create_directory_structure(data_root, split, save_dir):
    split_file = f'resisc45-{split}.txt'
    with open(os.path.join(save_dir, split_file), 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        class_name = '_'.join(l.split('_')[:-1])
        class_dir = os.path.join(save_dir, 'NWPU-RESISC45', class_name)
        os.makedirs(class_dir, exist_ok=True)
        # src_path = os.path.join(data_root, 'NWPU-RESISC45', l)
        src_path = os.path.join(data_root, class_name, l)
        dst_path = os.path.join(class_dir, l)
        print(f'{src_path} --> {dst_path}')
        os.symlink(src_path, dst_path)


DATA_ROOT = ''  # Path to the root directory of the dataset
SAVE_DIR = ''   # Path to the directory where the symlinks will be saved
for split in ['train', 'val', 'test']:
    create_directory_structure(DATA_ROOT, split, SAVE_DIR)
    
    