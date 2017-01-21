from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from constants import IMAGENET_SMALL_TF_DATADIR

slim = tf.contrib.slim

NUM_CLASSES = 1000

MIN_SIZE = 256

LABEL_OFFSET = 1

NAME = 'imagenet'

SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'height': 'Height of the image in pixels',
    'width': 'Width of the image in pixels',
    'edges': 'An edge map of the same size as the image.',
    'cartoon': 'An cartooned version of the image.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
}


def get_split(split_name, dataset_dir=IMAGENET_SMALL_TF_DATADIR, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      reader: The TensorFlow reader type.
    Returns:
      A `Dataset` namedtuple.
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    tf_record_pattern = os.path.join(dataset_dir, '%s-*' % split_name)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
        print('No files found for dataset at %s' % dataset_dir)
    else:
        print(data_files[:10])

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'height': slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
        'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', channels=1),
        'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', channels=3),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=data_files,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
    )
