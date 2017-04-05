import tensorflow as tf

import Dataset

slim = tf.contrib.slim
from constants import IMAGENET_TF_256_DATADIR, IMAGENET_SMALL_TF_DATADIR


class ImageNet(Dataset):
    SPLITS_TO_SIZES = {'train': 1281167, 'validation': 50000}

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying height and width.',
        'height': 'Height of the image in pixels',
        'width': 'Width of the image in pixels',
        'edges': 'An edge map of the same size as the image.',
        'cartoon': 'An cartooned version of the image.',
        'label': 'The label id of the image, integer between 0 and 999',
        'label_text': 'The text of the label.',
    }

    def __init__(self):
        super(ImageNet, self).__init__()
        self.data_dir = IMAGENET_TF_256_DATADIR
        self.file_pattern = '%s-*'
        self.num_classes = 1000
        self.label_offset = 1
        self.name = 'imagenet'

    def get_keys_to_features(self):
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
        return keys_to_features

    def get_items_to_handlers(self):
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
            'height': slim.tfexample_decoder.Tensor('image/height'),
            'width': slim.tfexample_decoder.Tensor('image/width'),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
            'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
            'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', channels=1),
            'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', channels=3),
        }
        return items_to_handlers

    def get_trainset(self):
        return self.get_split('train')

    def get_toon_train(self):
        return self.get_split('train', data_dir=IMAGENET_SMALL_TF_DATADIR)

    def get_testset(self):
        return self.get_split('validation')

    def get_toon_test(self):
        return self.get_split('validation', data_dir=IMAGENET_SMALL_TF_DATADIR)

    def num_train(self):
        return self.SPLITS_TO_SIZES['train']

    def num_test(self):
        return self.SPLITS_TO_SIZES['validation']
