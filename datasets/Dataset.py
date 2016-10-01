import h5py

from utils import im2float_array

IM_HEIGHT = 256
IM_WIDTH = 256


class Dataset:

    def __init__(self):
        """
            Base-class for all the datasets
        """
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Paths to data-files with randomized train and test images
        self.val_files = []
        self.train_files = []

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
            yield (im2float_array(train['X'][:]),
                   im2float_array(train['Y'][:]),
                   im2float_array(val['X'][:]),
                   im2float_array(val['Y'][:]))

    def get_sample(self, sample_size):
        """
            Returns a representative sample set of the training data.
        """
        train = h5py.File(self.train_files[0], 'r')
        sample_size = min(sample_size, len(train))
        return im2float_array(train['X'][:sample_size])
