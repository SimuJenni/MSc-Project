from datasets import download_and_convert_stl10, convert_voc2007, convert_imagenet, convert_imagenet_toon

from constants import *
from datasets import convert_voc2007_toon

download_and_convert_stl10.run()
convert_voc2007.run(VOC2007_TF_DATADIR, VOC2007_SRC_DIR)
convert_imagenet_toon.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TOON_TF_DATADIR)
convert_voc2007_toon.run(VOC2007_TOON_TF_DATADIR, VOC2007_SRC_DIR)
