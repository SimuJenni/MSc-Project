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
from scipy import misc

from constants import NUM_THREADS
from constants import IMAGENET_DATADIR, DATA_DIR


def process_image_files_batch(thread_index, ranges, name, out_dir, data):
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
        x_im = np.asscalar(misc.imread(x_fpath, mode='RGB'))
        y_im = np.asscalar(misc.imread(y_fpath, mode='RGB'))

        # construct the Example proto object
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_cartoon/encoded': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=x_im.astype("int64"))),
                    'image_cartoon/format': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_format])),
                    'image_original/encoded': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=y_im.astype("int64"))),
                    'image_original/format': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_format])),
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
    if not os.path.exists(os.path.join(out_dir, 'X')):
        os.makedirs(os.path.join(out_dir, 'X'))
    if not os.path.exists(os.path.join(out_dir, 'Y')):
        os.makedirs(os.path.join(out_dir, 'Y'))

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(data), NUM_THREADS + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (NUM_THREADS, ranges))
    sys.stdout.flush()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, out_dir, data)
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