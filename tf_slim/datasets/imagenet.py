# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the ImageNet ILSVRC 2012 Dataset plus some bounding boxes.
Some images have one or more bounding boxes associated with the label of the
image. See details here: http://image-net.org/download-bboxes
ImageNet is based upon WordNet 3.0. To uniquely identify a synset, we use
"WordNet ID" (wnid), which is a concatenation of POS ( i.e. part of speech )
and SYNSET OFFSET of WordNet. For more information, please refer to the
WordNet documentation[http://wordnet.princeton.edu/wordnet/documentation/].
"There are bounding boxes for over 3000 popular synsets available.
For each synset, there are on average 150 images with bounding boxes."
WARNING: Don't use for object detection, in this case all the bounding boxes
of the image belong to just one class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from six.moves import urllib

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'edges': 'An edge map of the same size as the image.',
    'cartoon': 'An cartooned version of the image.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
}

_NUM_CLASSES = 1001


def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string.
    Returns:
        labels_to_names: dictionary where keys are integers from to 1000
        and values are human-readable names.
    We retrieve a synset file, which contains a list of valid synset labels used
    by ILSVRC competition. There is one synset one per line, eg.
            #   n01440764
            #   n01443537
    We also retrieve a synset_to_human_file, which contains a mapping from synsets
    to human-readable names for every synset in Imagenet. These are stored in a
    tsv format, as follows:
            #   n02119247    black fox
            #   n02119359    silver fox
    We assign each synset (in alphabetical order) an integer, starting from 1
    (since 0 is reserved for the background class).
    Code is based on
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py#L463
    """

    # pylint: disable=g-line-too-long
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/'
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

    filename, _ = urllib.request.urlretrieve(synset_url)
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename, _ = urllib.request.urlretrieve(synset_to_human_url)
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def get_datafiles(split_name, dataset_dir):
    tf_record_pattern = os.path.join(dataset_dir, '%s-*' % split_name)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
        print('No files found for dataset at %s' % dataset_dir)
    return data_files


def get_split(split_name, dataset_dir, reader=None):
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
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
        'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format'),
        'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format'),
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
