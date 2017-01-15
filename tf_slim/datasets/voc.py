from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from constants import VOC2007_TF_DATADIR

import tensorflow as tf

slim = tf.contrib.slim

SPLITS_TO_SIZES = {'train': 2501, 'val': 2510, 'trainval': 5011, 'test': 4952}

NUM_CLASSES = 20

NAME = 'voc2007'

MIN_SIZE = 128

_FILE_PATTERN = 'voc2007_%s.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image.',
    'label': 'A single integer between 0 and 19 or -1 for unlabeled',
    'cartoon': 'A cartooned image.',
    'edges': 'An edge map.'
}


def get_split(split_name, dataset_dir=VOC2007_TF_DATADIR, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading voc2007.
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
        'class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
        'label': slim.tfexample_decoder.Tensor('class/label'),
        'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', channels=1),
        'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', channels=3),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)