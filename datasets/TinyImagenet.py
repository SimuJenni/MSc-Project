import glob
import os

from Dataset import Dataset

from constants import TINYIMAGENET_DATADIR, TINYIMAGENET_SRC_DIR

# For tiny-imagenet data preparation (100.000 train, 10.000 val)
NUM_SHARDS_TRAIN = 10
NUM_SHARDS_VAL = 10
IM_HEIGHT = 64
IM_WIDTH = 64


class TinyImagenet(Dataset):

    def __init__(self, resize=None):
        self.resize = resize
        self.src_data_dir = TINYIMAGENET_SRC_DIR
        self.data_dir = TINYIMAGENET_DATADIR
        self.train_dir = os.path.join(TINYIMAGENET_DATADIR, 'Train/')
        self.val_dir = os.path.join(TINYIMAGENET_DATADIR, 'Validation/')
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_tiny_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/%s*' % (self.val_dir, 'val'))
        self.train_files = glob.glob('%s/%s*' % (self.train_dir, 'train'))
        self.name = 'TinyImagenet'
        self.num_train = 100000
        self.num_val = 50000

    def process_tiny_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        from prepare_data_jpeg import process_dataset
        val_dir = os.path.join(self.src_data_dir, 'val/images/')
        train_dir = os.path.join(self.src_data_dir, 'train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, (IM_HEIGHT, IM_WIDTH))
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, (IM_HEIGHT, IM_WIDTH))


if __name__ == '__main__':
    data = TinyImagenet()

