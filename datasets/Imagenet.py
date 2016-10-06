import glob
import os

from Dataset import Dataset

from constants import IMAGENET_DATADIR, IMAGENET_SRC_DIR
from utils import montage

# For imagenet data preparation (12.000.000 train, 50.000 val)
NUM_SHARDS_TRAIN = 240
NUM_SHARDS_VAL = 10
IM_HEIGHT = 256
IM_WIDTH = 256


class Imagenet(Dataset):

    def __init__(self):
        self.src_data_dir = IMAGENET_SRC_DIR
        self.data_dir = IMAGENET_DATADIR
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/%s*' % (self.data_dir, 'validation'))
        self.train_files = glob.glob('%s/%s*' % (self.data_dir, 'train'))
        self.name = 'Imagenet'

    def process_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        from prepare_data import process_dataset
        val_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_val/')
        train_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_train/')
        print(val_dir)
        print(train_dir)

        process_dataset('validation', val_dir, self.data_dir, NUM_SHARDS_VAL, (IM_HEIGHT, IM_WIDTH))
        process_dataset('train', train_dir, self.data_dir, NUM_SHARDS_TRAIN, (IM_HEIGHT, IM_WIDTH))


if __name__ == '__main__':
    data = Imagenet()
    count = 1
    for train_x, train_y, _, _ in data.generator():
        montage(train_x[:100, :, :], 'Imagenet-X-{}'.format(count))
        montage(train_y[:100, :, :], 'Imagenet-Y-{}'.format(count))
        count += 1
