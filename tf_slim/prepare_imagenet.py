from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import threading
from datetime import datetime

import numpy as np
import tensorflow as tf

from constants import IMAGENET_DATADIR, DATA_DIR
from constants import NUM_THREADS


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_image_files_batch(thread_index, ranges, name, out_dir, data, coder):
    """Processes and saves list of images in 1 thread.

    Args:
      thread_index (int): Unique batch to run index is within [0, len(ranges)).
      ranges: List of pairs of integers specifying ranges of each batches to    analyze in parallel.
      name (str): Unique identifier specifying the data set
      out_dir: Diriectory for storing results
      data: List of strings; each string is a path to an image file
    """

    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    files_in_batch = np.arange(ranges[thread_index][0], ranges[thread_index][1], dtype=int)
    out_file = os.path.join(out_dir, "imagenet_toon_{}_{}.tfrecords".format(name, thread_index))
    writer = tf.python_io.TFRecordWriter(out_file)
    image_format = 'JPEG'

    for i in files_in_batch:
        x_fpath, y_fpath = data[i]
        with tf.gfile.FastGFile(x_fpath, 'r') as f:
            x_im_data = f.read()
        with tf.gfile.FastGFile(y_fpath, 'r') as f:
            y_im_data = f.read()
        x_im = coder.decode_jpeg(x_im_data)
        y_im = coder.decode_jpeg(y_im_data)

        # construct the Example proto object
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_cartoon/encoded': int64_feature(x_im),
                    'image_cartoon/format': bytes_feature(image_format),
                    'image_original/encoded': int64_feature(y_im),
                    'image_original/format': bytes_feature(image_format),
                }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
        if not counter % 500:
            print('[thread %d]: Processed %d of %d images in thread batch.' %
                  (thread_index, counter, num_files_in_thread))
            sys.stdout.flush()
        counter += 1


def process_image_files(name, out_dir, data):
    """Process and save list of images.

    Args:
      name (str): Unique identifier specifying the data set
      out_dir: Directory for storing results
      data: List of strings; each string is a path to an image file
    """

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(data), NUM_THREADS + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (NUM_THREADS, ranges))
    sys.stdout.flush()
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, out_dir, data, coder)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    for process in threads:
        process.join()

    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(data)))
    sys.stdout.flush()


def get_data(data_dir, name):
    x_dir = os.path.join(data_dir, '{}/X/'.format(name))
    y_dir = os.path.join(data_dir, '{}/Y/'.format(name))
    x_fpaths = [os.path.join(x_dir, fname) for fname in sorted(os.listdir(x_dir))]
    y_fpaths = [os.path.join(y_dir, fname) for fname in sorted(os.listdir(y_dir))]
    data = zip(x_fpaths, y_fpaths)
    shuffled_index = range(len(data))
    random.shuffle(shuffled_index)
    data = [data[i] for i in shuffled_index]

    print('Found %d JPEG files inside %s.' %
          (len(data), data_dir))
    return data


def process_dataset(name, src_dir, out_dir):
    """Process a complete data set and save it as JPEG files of given dimension.

    Args:
      name: string, unique identifier specifying the data set.
      src_dir: string, root path to the data set.
      out_dir: Directory for storing results
      num_shards: integer number of shards for this data set.
      im_dim: Tuple (height, width) defining the size of the images
    """
    data = get_data(src_dir, name)
    process_image_files(name, out_dir, data)


if __name__ == '__main__':
    out_dir = os.path.join(DATA_DIR, 'imagenet_toon_tf/')
    process_dataset('Validation', IMAGENET_DATADIR, out_dir)
