import tensorflow as tf

from Dataset import Dataset

slim = tf.contrib.slim
from constants import VOC2007_TF_DATADIR, VOC2007_TOON_TF_DATADIR


class VOC2007(Dataset):

    SPLITS_TO_SIZES = {'train': 2501, 'val': 2510, 'trainval': 5011, 'test': 4952}

    ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image.',
        'label': 'A single integer between 0 and 19 or -1 for unlabeled',
    }

    def __init__(self, cartoon_data_dir=VOC2007_TOON_TF_DATADIR):
        Dataset.__init__(self)
        self.data_dir = VOC2007_TF_DATADIR
        self.file_pattern = 'voc2007_%s.tfrecord'
        self.num_classes = 20
        self.name = 'VOC2007'
        self.is_multilabel = True
        self.toon_data_dir = cartoon_data_dir

    def get_keys_to_features(self):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/height': tf.FixedLenFeature((), tf.int64),
            'image/width': tf.FixedLenFeature((), tf.int64),
            'image/class/label': tf.FixedLenFeature([20], tf.int64, default_value=tf.zeros([20], dtype=tf.int64)),
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
            'edges': slim.tfexample_decoder.Image('edges/encoded', 'edges/format', channels=1),
            'cartoon': slim.tfexample_decoder.Image('cartoon/encoded', 'cartoon/format', channels=3),
        }
        return items_to_handlers

    def format_labels(self, labels):
        return labels

    def get_trainset(self):
        return self.get_split('trainval')

    def get_testset(self):
        return self.get_split('test')

    def get_num_train(self):
        return self.SPLITS_TO_SIZES['trainval']

    def get_num_test(self):
        return self.SPLITS_TO_SIZES['test']

    def get_toon_train(self):
        return self.get_split('trainval', data_dir=self.toon_data_dir)

    def get_num_train_toon(self):
        return self.SPLITS_TO_SIZES['trainval']