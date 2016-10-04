import os

DATA_DIR = '/Users/simujenni/MSc-Project/data/'
#DATA_DIR = '/data/cvg/simon/data/'

NUM_THREADS = 10

# Directories for cartooned datasets
CIFAR10_DATADIR = os.path.join(DATA_DIR, 'cifar-10-cartoon/')
TINYIMAGENET_DATADIR = os.path.join(DATA_DIR, 'tiny-imagenet-cartoon/')
IMAGENET_DATADIR = os.path.join(DATA_DIR, 'imagenet-cartoon/')

# Source directories for datasets
TINYIMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200-prepped/')
IMAGENET_SRC_DIR = '/data/cvg/imagenet/ILSVRC2012/'

