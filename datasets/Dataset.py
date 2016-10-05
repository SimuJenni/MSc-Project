import h5py
import gc
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

    def generator_train(self, batch_size):
        """
            A generator that loads training data for training.
            Yields a tuple (X_train, Y_train) where Xs are
            cartooned and Ys are original images
        """
        for i in range(0, len(self.train_files)):
            with h5py.File(self.train_files[i], 'r', driver='core', backing_store=False) as train:
                train_X = im2float_array(train['X'][:])
                train_Y = im2float_array(train['Y'][:])
                train.close()
            yield (trim2batchsize(train_X, batch_size),
                   trim2batchsize(train_Y, batch_size))

    def generator_test(self, batch_size):
        """
            A generator that loads validation data.
            Yields a tuple (X_test, Y_test) where Xs are
            cartooned and Ys are original images
        """
        for i in range(0, len(self.val_files_files)):
            with h5py.File(self.val_files[i], 'r', driver='core', backing_store=False) as val:
                test_X = im2float_array(val['X'][:])
                test_Y = im2float_array(val['Y'][:])
                val.close()
            yield (trim2batchsize(test_X, batch_size),
                   trim2batchsize(test_Y, batch_size))

    def generator(self, batch_size):
        """
            A generator that loads training and validation data for training.
            Yields a tuple (X_train, Y_train, X_test, Y_test) where Xs are
            cartooned and Ys are original images
        """
        num_val = len(self.val_files)
        for i in range(0, len(self.train_files)):
            with h5py.File(self.val_files[i % num_val], 'r', driver='core', backing_store=False) as val:
                test_X = im2float_array(val['X'][:])
                test_Y = im2float_array(val['Y'][:])
                val.close()
            with h5py.File(self.train_files[i], 'r', driver='core', backing_store=False) as train:
                train_X = im2float_array(train['X'][:])
                train_Y = im2float_array(train['Y'][:])
                train.close()
            yield (trim2batchsize(train_X, batch_size),
                   trim2batchsize(train_Y, batch_size),
                   trim2batchsize(test_X, batch_size),
                   trim2batchsize(test_Y, batch_size))

    def get_sample(self, sample_size):
        """
            Returns a representative sample set of the training data.
        """
        with h5py.File(self.train_files[0], 'r', driver='core', backing_store=False) as train:
            train_X = im2float_array(train['X'][:])
            sample_size = min(sample_size, train_X.shape[0])
            sample = train_X[:sample_size]
            train.close()
        return sample

    def train_batch_generator(self, batch_size):
        for X_train, Y_train in self.generator_train(batch_size):
            num_data = X_train.shape[0]
            gc.collect()
            for start in range(0, num_data, batch_size):
                yield (X_train[start:(start + batch_size)], Y_train[start:(start + batch_size)])


def trim2batchsize(samples, batch_size):
    num_samples = samples.shape[0]
    l = (num_samples // batch_size) * batch_size
    return samples[:l]
