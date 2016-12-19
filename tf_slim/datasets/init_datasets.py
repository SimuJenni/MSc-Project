from tf_slim.datasets import download_and_convert_cifar10, convert_imagenet, download_and_convert_stl10, convert_voc2007
from constants import *

# download_and_convert_cifar10.run(CIFAR10_TF_DATADIR)
# convert_imagenet.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TF_DATADIR)
# download_and_convert_stl10.run()
convert_voc2007.run(VOC2007_TF_DATADIR, VOC2007_SRC_DIR)
