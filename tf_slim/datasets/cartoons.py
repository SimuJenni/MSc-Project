from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from constants import CARTOON_TF_DATADIR

import tensorflow as tf

slim = tf.contrib.slim

NUM_SAMPLES = 12

MIN_SIZE = 256

NAME = 'cartoons'

_FILE_NAME = 'cartoons.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'cartoon': 'A [256 x 256 x 3] cartooned image.',
    'edges': 'A [256 x 256 x 1] edge map.'
}


def get_data(dataset_dir=CARTOON_TF_DATADIR, reader=None):
    """Gets a dataset tuple with instructions for reading cartoons.
    Args:
      dataset_dir: The base directory of the dataset sources.
      reader: The TensorFlow reader type.
    Returns:
      A `Dataset` namedtuple.
    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """

    file_name = os.path.join(dataset_dir, _FILE_NAME)

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
        'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', shape=[256, 256, 1], channels=1),
        'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', shape=[256, 256, 3], channels=3),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_name,
        reader=reader,
        decoder=decoder,
        num_samples=NUM_SAMPLES,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
