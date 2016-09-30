import os

#DATA_DIR = '/Users/simujenni/MSc-Project/data/'
DATA_DIR = '/data/cvg/simon/data/'

NUM_THREADS = 10

# Directories for cartooned datasets
CIFAR10_DATADIR = os.path.join(DATA_DIR, 'cifar-10-cartoon/')
TINYIMAGENET_DATADIR = os.path.join(DATA_DIR, 'tiny-imagenet-cartoon/')

# Source directories for datasets
#TINYIMAGENET_SRC_DIR = '/Users/simujenni/Data/tiny-imagenet-200-prepped'
TINYIMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200-prepped/')

