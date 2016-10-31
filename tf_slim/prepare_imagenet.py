# load up some dataset. Could be anything but skdata is convenient.
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os
from PIL import Image

from constants import IMAGENET_DATADIR


train_X_dir = os.path.join(IMAGENET_DATADIR, 'Train/X/')
train_Y_dir = os.path.join(IMAGENET_DATADIR, 'Train/X/')
train_data = zip(sorted(os.listdir(train_X_dir)), sorted(os.listdir(train_Y_dir)))
trIdx = range(len(train_data))

# randomly shuffle data
np.random.shuffle(trIdx)
writer = tf.python_io.TFRecordWriter("imagenet-toon.tfrecords")

# iterate over each example
for example_idx in tqdm(trIdx):
    features = data.all_vectors[example_idx]
    label = data.all_labels[example_idx]

    # construct the Example proto object
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(
                int64_list=tf.train.Int64List(value=features.astype("int64"))),
    }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)