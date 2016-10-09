import os

NUM_THREADS = 10

# Directories for data, images and models
DATA_DIR = '/Users/simujenni/MSc-Project/data/'
#DATA_DIR = '/data/cvg/simon/data/'
MODEL_DIR = os.path.join(DATA_DIR, 'models/')
IMG_DIR = os.path.join(DATA_DIR, 'img/')
LOG_DIR = os.path.join(DATA_DIR, 'logs/')

# Directories for cartooned datasets
CIFAR10_DATADIR = os.path.join(DATA_DIR, 'cifar-10-cartoon/')
TINYIMAGENET_DATADIR = os.path.join(DATA_DIR, 'tiny-imagenet-toon/')
IMAGENET_DATADIR = os.path.join(DATA_DIR, 'imagenet-toon/')

# Source directories for datasets
TINYIMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200-prepped/')
IMAGENET_SRC_DIR = '/data/cvg/imagenet/ILSVRC2012/'

