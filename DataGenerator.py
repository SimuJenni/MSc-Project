from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
import os
import threading

from keras import backend as K


def X2X_Y2Y(x, y):
    return x, y


def X2X_X2Y(x, y):
    return x, x


def Y2X_Y2Y(x, y):
    return y, y


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, scale=True):
    from PIL import Image
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img):
    # image has dim_ordering (height, width, channel)
    x = (np.asarray(img, dtype='float32') / 255. - 0.5) * 2.0
    if len(x.shape) == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    return x


def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    # Arguments
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 horizontal_flip=False,
                 dim_ordering='default'):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.horizontal_flip = horizontal_flip

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg', xy_fun=X2X_Y2Y):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, xy_fun=xy_fun)

    def flow_from_directory(self, directory,
                            target_size=(192, 192), color_mode='rgb',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg', xy_fun=X2X_Y2Y):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, xy_fun=xy_fun)

    def random_transform(self, x, y):
        img_col_index = self.col_index - 1

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        return x, y


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, X, Y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg', xy_fun=X2X_Y2Y):
        if Y is not None and len(X) != len(Y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(Y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.Y = Y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.xy_fun = xy_fun
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y = np.zeros(tuple([current_batch_size] + list(self.Y.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            y = self.Y[j]
            x, y = self.image_data_generator.random_transform(x, y)
            x, y = self.xy_fun(x, y)
            batch_x[i] = x
            batch_y[i] = y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y


class DirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(192, 192), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', xy_fun=X2X_Y2Y):
        self.X_dir = os.path.join(directory, 'X/')
        self.Y_dir = os.path.join(directory, 'Y/')
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = self.target_size + (1,)
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.xy_fun = xy_fun

        self.nb_sample = 0
        self.filenames = []

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        for fname_x, fname_y in zip(sorted(os.listdir(self.X_dir)), sorted(os.listdir(self.Y_dir))):
            if not fname_x == fname_y:
                raise Exception("{} and {} are not of the same underlying image!".format(fname_x, fname_y))

            is_valid = False
            for extension in white_list_formats:
                if fname_x.lower().endswith('.' + extension) and fname_y.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.filenames.append((os.path.join(self.X_dir, fname_x), os.path.join(self.Y_dir, fname_y)))
                self.nb_sample += 1

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname_x, fname_y = self.filenames[j]
            img_x = load_img(os.path.join(self.X_dir, fname_x), grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img_x)
            img_y = load_img(os.path.join(self.Y_dir, fname_y), grayscale=grayscale, target_size=self.target_size)
            y = img_to_array(img_y)
            x, y = self.image_data_generator.random_transform(x, y)
            x, y = self.xy_fun(x, y)
            batch_x[i] = x
            batch_y[i] = y
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y
