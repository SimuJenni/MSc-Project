from datasets import download_and_convert_stl10, convert_voc2007, convert_imagenet

from constants import *
from datasets import convert_imagenet_toon

# download_and_convert_stl10.run()
# convert_voc2007.run(VOC2007_TF_DATADIR, VOC2007_SRC_DIR)
# convert_imagenet_toon.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_SMALL_TF_DATADIR)
convert_imagenet_toon.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TF_256_TOON_DATADIR, im_size=(256, 256))
