from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from constants import IMAGENET_DATADIR

import tensorflow as tf


slim = tf.contrib.slim

_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image_cartoon': 'A cartooned image',
    'image_original': 'The ground truth image',
}

_NUM_CLASSES = 1001


def get_split(split_name='Train', dataset_dir=IMAGENET_DATADIR, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.
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
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    X_dir = os.path.join(dataset_dir, '%s/X/' % split_name)
    Y_dir = os.path.join(dataset_dir, '%s/Y/' % split_name)

    data_sources = zip(sorted(os.listdir(X_dir)), sorted(os.listdir(Y_dir)))

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image_cartoon/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image_cartoon/format': tf.FixedLenFeature(
            (), tf.string, default_value='JPEG'),
        'image_original/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image_original/format': tf.FixedLenFeature(
            (), tf.string, default_value='JPEG'),
    }

    items_to_handlers = {
        'image_cartoon': slim.tfexample_decoder.Image('image_cartoon/encoded', 'image_cartoon/format'),
        'image_original': slim.tfexample_decoder.Image('image_original/encoded', 'image_original/format'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES)



