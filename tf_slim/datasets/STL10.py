import tensorflow as tf

from Dataset import Dataset

slim = tf.contrib.slim
from constants import STL10_TF_DATADIR


class STL10(Dataset):

    SPLITS_TO_SIZES = {'train_unlabeled': 100000, 'train': 5000, 'test': 8000, 'train_fold_0': 4000,
                       'train_fold_1': 4000,
                       'train_fold_2': 4000, 'train_fold_3': 4000, 'train_fold_4': 4000, 'train_fold_5': 4000,
                       'train_fold_6': 4000, 'train_fold_7': 4000, 'train_fold_8': 4000, 'train_fold_9': 4000,
                       'test_fold_0': 1000, 'test_fold_1': 1000, 'test_fold_2': 1000, 'test_fold_3': 1000,
                       'test_fold_4': 1000, 'test_fold_5': 1000, 'test_fold_6': 1000, 'test_fold_7': 1000,
                       'test_fold_8': 1000, 'test_fold_9': 1000
                       }

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A [96 x 96 x 3] color image.',
        'label': 'A single integer between 0 and 9 or -1 for unlabeled',
        'cartoon': 'A [96 x 96 x 3] cartooned image.',
        'edges': 'A [96 x 96 x 1] edge map.'
    }

    def __init__(self):
        Dataset.__init__(self)
        self.data_dir = STL10_TF_DATADIR
        self.file_pattern = 'stl10_%s.tfrecord'
        self.num_classes = 10
        self.name = 'STL10'

    def get_keys_to_features(self):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/height': tf.FixedLenFeature((), tf.int64, default_value=96),
            'image/width': tf.FixedLenFeature((), tf.int64, default_value=96),
            'class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'edges/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'edges/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'cartoon/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'cartoon/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        }
        return keys_to_features

    def get_items_to_handlers(self):
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', shape=[96, 96, 3], channels=3),
            'height': slim.tfexample_decoder.Tensor('image/height'),
            'width': slim.tfexample_decoder.Tensor('image/width'),
            'label': slim.tfexample_decoder.Tensor('class/label'),
            'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', shape=[96, 96, 1], channels=1),
            'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', shape=[96, 96, 3], channels=3),
        }
        return items_to_handlers

    def get_trainset(self):
        return self.get_split('train')

    def get_toon_train(self):
        return self.get_split('train_unlabeled')

    def get_testset(self):
        return self.get_split('test')

    def get_train_fold_id(self, fold_idx):
        return 'train_fold_{}'.format(fold_idx)

    def get_test_fold_id(self, fold_idx):
        return 'test_fold_{}'.format(fold_idx)

    def get_num_train(self):
        return self.SPLITS_TO_SIZES['train_unlabeled']

    def get_num_test(self):
        return self.SPLITS_TO_SIZES['test']
