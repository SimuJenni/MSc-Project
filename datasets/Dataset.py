import h5py
import gc
from utils import im2float
from scipy.misc import imresize
import numpy as np

IM_HEIGHT = 256
IM_WIDTH = 256


class Dataset:
    def __init__(self, resize=None):
        """
            Base-class for all the datasets
        """
        self.dims = (IM_HEIGHT, IM_HEIGHT, 3)
        # Paths to data-files with randomized train and test images
        self.val_files = []
        self.train_files = []
        self.resize = resize

    def generator_train_h5(self, batch_size):
        """
            A generator that loads training data for training.
            Yields a tuple (X_train, Y_train) where Xs are
            cartooned and Ys are original images
        """
        for i in range(0, len(self.train_files)):
            with h5py.File(self.train_files[i], 'r') as train:
                train_X = self.preprocess(train['X'][:])
                train_Y = self.preprocess(train['Y'][:])
            yield (trim2batchsize(train_X, batch_size),
                   trim2batchsize(train_Y, batch_size))

    def generator_test_h5(self, batch_size):
        """
            A generator that loads validation data.
            Yields a tuple (X_test, Y_test) where Xs are
            cartooned and Ys are original images
        """
        for i in range(0, len(self.val_files)):
            with h5py.File(self.val_files[i], 'r') as val:
                test_X = self.preprocess(val['X'][:])
                test_Y = self.preprocess(val['Y'][:])
            yield (trim2batchsize(test_X, batch_size),
                   trim2batchsize(test_Y, batch_size))

    def generator_h5(self, batch_size):
        """
            A generator that loads training and validation data for training.
            Yields a tuple (X_train, Y_train, X_test, Y_test) where Xs are
            cartooned and Ys are original images
        """
        num_val = len(self.val_files)
        for i in range(0, len(self.train_files)):
            with h5py.File(self.val_files[i % num_val], 'r') as val:
                test_X = self.preprocess(val['X'][:])
                test_Y = self.preprocess(val['Y'][:])
            with h5py.File(self.train_files[i], 'r') as train:
                train_X = self.preprocess(train['X'][:])
                train_Y = self.preprocess(train['Y'][:])
            yield (trim2batchsize(train_X, batch_size),
                   trim2batchsize(train_Y, batch_size),
                   trim2batchsize(test_X, batch_size),
                   trim2batchsize(test_Y, batch_size))

    def get_sample_h5(self, sample_size):
        """
            Returns a representative sample set of the training data.
        """
        with h5py.File(self.train_files[0], 'r') as train:
            train_X = self.preprocess(train['X'][:])
            sample_size = min(sample_size, train_X.shape[0])
            sample = train_X[:sample_size]
        return sample

    def get_sample_dir(self, sample_size):
        """
            Returns a sample set of the training data of given size.
        """
        from PIL import Image
        if self.resize:
            sample = np.zeros((sample_size, ) + self.resize)
        else:
            sample = np.zeros((sample_size, ) + self.dims)
        for i in range(sample_size):
            img = Image.open(self.train_files[i])
            if self.resize:
                img = imresize(img, self.resize)
            sample[i, :, :] = im2float(img)
        return sample

    def preprocess(self, X):
        num_im = X.shape[0]
        if self.resize:
            X_res = np.zeros((num_im,)+self.resize)
            for i in range(num_im):
                X_res[i, :, :] = im2float(imresize(X[i, :, :], self.resize))
        else:
            X_res = im2float(X)
        # Rescale values from [0,1] to [-1,1]
        return (X_res-0.5)*2


def trim2batchsize(samples, batch_size):
    num_samples = samples.shape[0]
    l = (num_samples // batch_size) * batch_size
    return samples[:l]
