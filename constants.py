import os

NUM_THREADS = 16

# Directories for data, images and models
#DATA_DIR = '/Users/simujenni/MSc-Project/data/'
DATA_DIR = '/data/cvg/simon/data/'
MODEL_DIR = os.path.join(DATA_DIR, 'models/')
IMG_DIR = os.path.join(DATA_DIR, 'img/')
LOG_DIR = os.path.join(DATA_DIR, 'logs/')

# Directories for cartooned datasets
CIFAR10_TOON_DATADIR = os.path.join(DATA_DIR, 'cifar-10-cartoon/')
TINYIMAGENET_TOON_DATADIR = os.path.join(DATA_DIR, 'tiny-imagenet-toon/')
IMAGENET_TOON_DATADIR = os.path.join(DATA_DIR, 'imagenet-toon/')
IMAGENET_TF_DATADIR = os.path.join(DATA_DIR, 'imagenet-TFRecords/')
VOC2007_TF_DATADIR = os.path.join(DATA_DIR, 'voc2007-TFRecords/')
CIFAR10_TF_DATADIR = os.path.join(DATA_DIR, 'cifar-10-TFRecords/')
STL10_TF_DATADIR = os.path.join(DATA_DIR, 'stl-10-TFRecords/')

# Source directories for datasets
STL10_DATADIR = os.path.join(DATA_DIR, 'stl-10/')
CIFAR10_DATADIR = os.path.join(DATA_DIR, 'cifar-10/')
TINYIMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200-prepped/')
VOC2007_SRC_DIR = os.path.join(DATA_DIR, 'VOCdevkit/')
IMAGENET_SRC_DIR = '/data/cvg/imagenet/ILSVRC2012/'
IMAGENET_TRAIN_DIR = os.path.join(IMAGENET_SRC_DIR, 'ILSVRC2012_img_train/')
IMAGENET_VAL_DIR = os.path.join(DATA_DIR, 'ILSVRC2012_img_val/')
