from tf_slim.datasets import download_and_convert_cifar10
from constants import DATA_DIR
import os

download_and_convert_cifar10.run(os.path.join(DATA_DIR, 'cifar-10-TFRecords/'))