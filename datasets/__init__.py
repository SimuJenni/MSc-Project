import glob

from Dataset import Dataset
from constants import *
from keras.datasets import cifar10
import h5py


class Imagenet(Dataset):

    def __init__(self, resize=None, num_train=1200000, target_size=(196, 196)):
        self.resize = resize
        self.src_data_dir = IMAGENET_SRC_DIR
        self.data_dir = IMAGENET_DATADIR
        self.train_dir = os.path.join(IMAGENET_DATADIR, 'Train/')
        self.val_dir = os.path.join(IMAGENET_DATADIR, 'Validation/')
        self.target_size = target_size
        self.dims = target_size + (3,)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/X/%s*' % (self.val_dir, 'val'))
        self.train_files = glob.glob('%s/X/%s*' % (self.train_dir, 'train'))
        if num_train:
            self.train_files = self.train_files[:num_train]
        self.name = 'Imagenet_numTrain_{}_targetSize_{}'.format(num_train, target_size)
        self.num_train = num_train
        self.num_val = 50000

    def process_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        from prepare_data_jpeg import process_dataset
        val_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_val/')
        train_dir = os.path.join(self.src_data_dir, 'ILSVRC2012_img_train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, (192, 192))
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, (192, 192))


class TinyImagenet(Dataset):

    def __init__(self, resize=None, num_train=100000, target_size=(64, 64)):
        self.resize = resize
        self.src_data_dir = TINYIMAGENET_SRC_DIR
        self.data_dir = TINYIMAGENET_DATADIR
        self.train_dir = os.path.join(TINYIMAGENET_DATADIR, 'Train/')
        self.val_dir = os.path.join(TINYIMAGENET_DATADIR, 'Validation/')
        self.target_size = target_size
        self.dims = target_size + (3,)
        # Check if dataset has already been preprocessed
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.process_tiny_imagenet()
        # Paths to data-files with randomized train and test images
        self.val_files = glob.glob('%s/%s*' % (self.val_dir, 'val'))
        self.train_files = glob.glob('%s/%s*' % (self.train_dir, 'train'))
        self.name = 'TinyImagenet_numTrain_{}_targetSize_{}'.format(num_train, target_size)
        self.num_train = num_train
        self.num_val = 50000

    def process_tiny_imagenet(self):
        """
            Pre-processes the source data and stores in batches for training
        """
        from prepare_data_jpeg import process_dataset
        val_dir = os.path.join(self.src_data_dir, 'val/images/')
        train_dir = os.path.join(self.src_data_dir, 'train/')
        print('Preparing images in: {}'.format(val_dir))
        process_dataset('val', val_dir, self.val_dir, (64, 64))
        print('Preparing images in: {}'.format(train_dir))
        process_dataset('train', train_dir, self.train_dir, (64, 64))


def cifar10_cartoon_data():
    # First check if dir exists and create if not
    if not os.path.exists(CIFAR10_DATADIR):
        os.makedirs(CIFAR10_DATADIR)

    file_name = 'dataset.hdf5'
    data_path = os.path.join(CIFAR10_DATADIR, file_name)

    # Cartooned data already exist -> load and return it
    if os.path.isfile(data_path):
        with h5py.File(data_path, 'r') as data:
            return (data['X_train'][:], data['Y_train'][:]), (data['X_test'][:], data['Y_test'][:])

    # Otherwise get the data set and process it
    from cartooning import process_data
    (Y_train, _), (Y_test, _) = cifar10.load_data()
    X_train = process_data(Y_train)
    X_test = process_data(Y_test)

    # Save the processed data and return
    with h5py.File(data_path, 'w') as data:
        data.create_dataset('X_train', data=X_train)
        data.create_dataset('Y_train', data=Y_train)
        data.create_dataset('X_test', data=X_test)
        data.create_dataset('Y_test', data=Y_test)
        return (data['X_train'][:], data['Y_train'][:]), (data['X_test'][:], data['Y_test'][:])