from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

slim = tf.contrib.slim


class Dataset:
    SPLITS_TO_SIZES = {}
    ITEMS_TO_DESCRIPTIONS = {}

    def __init__(self):
        self.reader = tf.TFRecordReader
        self.label_offset = 0
        self.data_dir = None
        self.num_classes = None
        self.file_pattern = '%s-*'
        self.is_multilabel = False

    def get_trainset(self):
        pass

    def get_toon_train(self):
        pass

    def get_testset(self):
        pass

    def get_toon_test(self):
        return self.get_testset()

    def get_num_train(self):
        pass

    def get_num_train_toon(self):
        pass

    def get_num_test(self):
        pass

    def get_num_dataset(self, id):
        return self.SPLITS_TO_SIZES[id]

    def get_keys_to_features(self):
        pass

    def get_items_to_handlers(self):
        pass

    def get_split_size(self, split_name):
        return self.SPLITS_TO_SIZES[split_name]

    def format_labels(self, labels):
        return slim.one_hot_encoding(labels, self.num_classes)

    def get_split(self, split_name, data_dir=None):
        """Gets a dataset tuple with instructions for reading ImageNet.
        Args:
          split_name: A train/test split name.
          dataset_dir: The base directory of the dataset sources.
        Returns:
          A `Dataset` namedtuple.
        Raises:
          ValueError: if `split_name` is not a valid train/test split.
        """
        if split_name not in self.SPLITS_TO_SIZES:
            raise ValueError('split name %s was not recognized.' % split_name)

        if not data_dir:
            data_dir = self.data_dir

        tf_record_pattern = os.path.join(data_dir, self.file_pattern % split_name)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset at %s' % data_dir)

        # Build the decoder
        keys_to_features = self.get_keys_to_features()
        items_to_handlers = self.get_items_to_handlers()
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        return slim.dataset.Dataset(
            data_sources=data_files,
            reader=self.reader,
            decoder=decoder,
            num_samples=self.SPLITS_TO_SIZES[split_name],
            items_to_descriptions=self.ITEMS_TO_DESCRIPTIONS
        )
