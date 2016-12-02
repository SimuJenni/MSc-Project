from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from constants import CIFAR10_TF_DATADIR

import tensorflow as tf

slim = tf.contrib.slim

SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}

NUM_CLASSES = 10

MIN_SIZE = 32

NAME = 'cifar10'

_FILE_PATTERN = 'cifar10_%s.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
    'cartoon': 'A [32 x 32 x 3] cartooned image.',
    'edges': 'A [32 x 32 x 1] edge map.'
}


def get_split(split_name, dataset_dir=CIFAR10_TF_DATADIR, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.
    Returns:
      A `Dataset` namedtuple.
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=32),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=32),
        'class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', shape=[32, 32, 3], channels=3),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'label': slim.tfexample_decoder.Tensor('class/label'),
        'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', shape=[32, 32, 1], channels=1),
        'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', shape=[32, 32, 3], channels=3),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
