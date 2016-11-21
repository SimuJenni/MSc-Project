from tf_slim.datasets import convert_imagenet, download_and_convert_stl10
from constants import IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TF_DATADIR

convert_imagenet.run(IMAGENET_TRAIN_DIR, IMAGENET_VAL_DIR, IMAGENET_TF_DATADIR)
download_and_convert_stl10.run()