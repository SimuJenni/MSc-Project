import glob
import os

import h5py

from constants import TINYIMAGENET_DATADIR, TINYIMAGENET_SRC_DIR
from prepare_data import process_dataset
from utils import montage

# For tiny-imagenet data preparation (100.000 train, 10.000 val)
NUM_SHARDS_TRAIN = 10
NUM_SHARDS_VAL = 10
IM_HEIGHT = 64
IM_WIDTH = 64


class TinyImagenet:

    def __init__(self):
        self.src_data_dir = TINYIMAGENET_SRC_DIR
        self.data_dir = TINYIMAGENET_DATADIR
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_tiny_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/%s*' % (self.data_dir, 'validation'))
        self.train_files = glob.glob('%s/%s*' % (self.data_dir, 'train'))

    def get_dims(self):
        """
            Returns the dimensions of the images as 3D array (height, width, channel)
        """
        return self.dims

    def generator(self):
        """
            A generator that loads training and validation data for training.
            Yields a tuple (X_train, Y_train, X_test, Y_test) where Xs are
            cartooned and Ys are original images
        """
        num_val = len(self.val_files)
        for i in range(0, len(self.train_files)):
            val = h5py.File(self.val_files[i % num_val], 'r')
            train = h5py.File(self.train_files[i], 'r')
            yield (train['X'][:], train['Y'][:], val['X'][:], val['Y'][:])

    def get_sample(self, sample_size):
        """
            Returns a representative sample set of the training data.
        """
        train = h5py.File(self.train_files[0], 'r')
        sample_size = min(sample_size, len(train))
        return train['X'][:sample_size]

    def process_tiny_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        val_dir = os.path.join(self.src_data_dir, 'val/images/')
        train_dir = os.path.join(self.src_data_dir, 'train/')
        print(val_dir)
        print(train_dir)

        process_dataset('validation', val_dir, self.data_dir, NUM_SHARDS_VAL, (IM_HEIGHT, IM_WIDTH))
        process_dataset('train', train_dir, self.data_dir, NUM_SHARDS_TRAIN, (IM_HEIGHT, IM_WIDTH))


if __name__ == '__main__':
    data = TinyImagenet()
    count = 1
    for train_x, train_y, _, _ in data.generator():
        montage(train_x[:100, :, :] / 255., 'TiniImagenet-X-{}'.format(count))
        montage(train_y[:100, :, :] / 255., 'TiniImagenet-Y-{}'.format(count))
        count += 1
