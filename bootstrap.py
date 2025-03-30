#!/usr/bin/env python
import os
import glob
import shutil
import urllib.request
import tarfile
import numpy as np
from scipy.io import loadmat

def download_file(url, dest=None):
    if not dest:
        filename = url.split('/')[-1]
        dest = os.path.join('datasets', 'flowers', filename)
    urllib.request.urlretrieve(url, dest)

# Create dataset directory and download files
data_dir = os.path.join('datasets', 'flowers')
os.makedirs(data_dir, exist_ok=True)

# Check if data needs to be downloaded
required_files = [
    os.path.join(data_dir, 'imagelabels.mat'),
    os.path.join(data_dir, 'setid.mat')
]
print("Checking for required files...")

if not all(os.path.exists(f) for f in required_files):
    print("Downloading images...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')

    # Extract images
    with tarfile.open(os.path.join(data_dir, '102flowers.tgz'), 'r:gz') as tar:
        tar.extractall(path=data_dir)

    print("Downloading labels...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

    print("Downloading splits...")
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')

setid = loadmat(os.path.join(data_dir, 'setid.mat'))
idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

image_labels = loadmat(os.path.join(data_dir, 'imagelabels.mat'))['labels'][0]
image_labels -= 1

jpg_files = sorted(glob.glob(os.path.join(data_dir, 'jpg', '*.jpg')))
relative_paths = [os.path.relpath(f, data_dir) for f in jpg_files]
labels = np.array(list(zip(relative_paths, image_labels)), dtype=object)


if not os.path.exists(os.path.join(data_dir, 'jpg', '1')):
    print("Organizing training images into class directories...")
    for idx in idx_train:
        orig_path, label = labels[idx]
        class_num = int(label) + 1
        
        class_dir = os.path.join(data_dir, 'jpg', str(class_num))
        os.makedirs(class_dir, exist_ok=True)
        
        src = os.path.join(data_dir, orig_path)
        dst = os.path.join(class_dir, os.path.basename(orig_path))
        shutil.move(src, dst)

        new_path = os.path.join(str(class_num), os.path.basename(orig_path))
        # append "jpg" to the new path
        new_path = os.path.join('jpg', new_path)
        labels[idx] = (new_path, label)
      

def write_split(fname, indices):
    with open(os.path.join(data_dir, fname), 'w') as f:
        for path, label in labels[indices]:
            #print(path, label)
            f.write(f'{path} {label}\n')

np.random.seed(777)
write_split('train.txt', np.random.permutation(idx_train))
write_split('test.txt', np.random.permutation(idx_test))
write_split('valid.txt', np.random.permutation(idx_valid))

print("Dataset preparation complete!")
