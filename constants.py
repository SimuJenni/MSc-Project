import os

NUM_THREADS = 16

# Directories for data, images and models
DATA_DIR = '/Data/'
MODEL_DIR = os.path.join(DATA_DIR, 'models/')
LOG_DIR = os.path.join(DATA_DIR, 'Logs/ToonNet')

# Directories for cartooned datasets
IMAGENET_TF_256_DATADIR = os.path.join(DATA_DIR, 'TF_records/imagenet-TFRecords/')
IMAGENET_SMALL_TF_DATADIR = os.path.join(DATA_DIR, 'TF_records/imagenet-small-TFRecords/')
VOC2007_TF_DATADIR = os.path.join(DATA_DIR, 'TF_records/voc2007-TFRecords/')
STL10_TF_DATADIR = os.path.join(DATA_DIR, 'TF_records/stl-10-TFRecords/')

# Source directories for datasets
STL10_DATADIR = os.path.join(DATA_DIR, 'Datasets/stl-10/')
VOC2007_SRC_DIR = os.path.join(DATA_DIR, 'Datasets/VOCdevkit/')
IMAGENET_SRC_DIR = os.path.join(DATA_DIR, 'Datasets/imagenet/ILSVRC2012/')
IMAGENET_TRAIN_DIR = os.path.join(IMAGENET_SRC_DIR, 'ILSVRC2012_img_train/')
IMAGENET_VAL_DIR = os.path.join(DATA_DIR, 'ILSVRC2012_img_val/')
