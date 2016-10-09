import glob
import os

from Dataset import Dataset

from constants import IMAGENET_DATADIR, IMAGENET_SRC_DIR

# For imagenet data preparation (12.000.000 train, 50.000 val)
IM_HEIGHT = 192
IM_WIDTH = 192

class Imagenet(Dataset):

    def __init__(self, resize=None):
        self.resize = resize
        self.src_data_dir = IMAGENET_SRC_DIR
        self.data_dir = IMAGENET_DATADIR
        self.train_dir = os.path.join(IMAGENET_DATADIR, 'Train/')
        self.val_dir = os.path.join(IMAGENET_DATADIR, 'Validation/')
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/%s*' % (self.val_dir, 'val'))
        self.train_files = glob.glob('%s/%s*' % (self.train_dir, 'train'))
        self.name = 'Imagenet'
        self.num_train = 1200000
        self.num_val = 50000

    def process_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        from prepare_data_jpeg import process_dataset
        val_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_val/')
        train_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, (IM_HEIGHT, IM_WIDTH))
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, (IM_HEIGHT, IM_WIDTH))


if __name__ == '__main__':
    data = Imagenet()
