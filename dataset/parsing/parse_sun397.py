import os, pdb

ROOT = ""       # Path to SUN397 dataset
SAVE_DIR = ""   # Path to save directory
TRAIN_SAVE_DIR = os.path.join(SAVE_DIR, "train")
TEST_SAVE_DIR = os.path.join(SAVE_DIR, "test")

os.makedirs(TRAIN_SAVE_DIR, exist_ok=True)
os.makedirs(TEST_SAVE_DIR, exist_ok=True)

train_list_path = os.path.join(SAVE_DIR, "Training_01.txt")
test_list_path = os.path.join(SAVE_DIR, "Testing_01.txt")

with open(train_list_path, 'r') as f_train:
    train_paths = f_train.readlines()

train_paths = [path.strip() for path in train_paths]

with open(test_list_path, 'r') as f_test:
    test_paths = f_test.readlines()

test_paths = [path.strip() for path in test_paths]

for path in train_paths:
    # Create full path for symlink (train)
    symlink_path = os.path.join(ROOT, os.path.relpath(path, '/'))
    print(f"Symlink path: {symlink_path}")

    nested_paths = path.strip().split('/')[1:]
    image_class = '_'.join(nested_paths[:-1])
    rel_path = os.path.join(image_class, nested_paths[-1])
    dest_path = os.path.join(TRAIN_SAVE_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        os.symlink(symlink_path, dest_path)
    except: 
        pdb.set_trace()
    print(f"Symlink created: {symlink_path}")

for path in test_paths:
    # Create full path for symlink (test)
    symlink_path = os.path.join(ROOT, os.path.relpath(path, '/'))
    nested_paths = path.strip().split('/')[1:]
    image_class = '_'.join(nested_paths[:-1])
    rel_path = os.path.join(image_class, nested_paths[-1])
    dest_path = os.path.join(TEST_SAVE_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Create symlink
    try:
        os.symlink(symlink_path, dest_path)
    except: 
        pdb.set_trace()
        
    print(f"Symlink created: {symlink_path}")

