from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import sys
import threading
from datetime import datetime

import h5py
import numpy as np
from scipy import misc

from cartooning import cartoonify_bilateral
from constants import DATA_DIR, NUM_THREADS
from utils import imcrop_tosquare


def process_image(filename, im_dim):
    """Process a single image

    Args:
        filename (str): Full path to image file
        im_dim: Tuple (height, width) defining the size of the images

    Returns:
        np.array of processed image data
    """
    # Read the image file.
    image_data = misc.imread(filename, mode='RGB')
    image_data = imcrop_tosquare(image_data)
    image_data = misc.imresize(image_data, im_dim)
    image_cartoon = cartoonify_bilateral(image_data)
    return (image_cartoon, image_data)


def process_image_files_batch(thread_index, ranges, name, out_dir, filenames, num_shards, im_dim):
    """Processes and saves list of images in 1 thread.

    Args:
      thread_index (int): Unique batch to run index is within [0, len(ranges)).
      ranges: List of pairs of integers specifying ranges of each batches to    analyze in parallel.
      name (str): Unique identifier specifying the data set
      out_dir: Diriectory for storing results
      filenames: List of strings; each string is a path to an image file
      num_shards (int): Number of shards for this data set.
      im_dim: Tuple (height, width) defining the size of the images
    """

    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(out_dir, output_filename)
        with h5py.File(output_file, 'w') as out_file:
            shard_counter = 0
            files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
            X = out_file.create_dataset('X', shape=(len(files_in_shard), im_dim[0], im_dim[1], 3), dtype=np.uint8)
            Y = out_file.create_dataset('Y', shape=(len(files_in_shard), im_dim[0], im_dim[1], 3), dtype=np.uint8)
            for i, j in enumerate(files_in_shard):
                filename = filenames[j]
                image_data, image_cartoon = process_image(filename, im_dim)
                X[i, :, :, :] = image_data
                Y[i, :, :, :] = image_cartoon
                shard_counter += 1
                counter += 1

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()

        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def process_image_files(name, out_dir, filenames, num_shards, im_dim):
    """Process and save list of images.

    Args:
      name (str): Unique identifier specifying the data set
      out_dir: Diriectory for storing results
      filenames: List of strings; each string is a path to an image file
      num_shards (int): Number of shards for this data set.
      im_dim: Tuple (height, width) defining the size of the images
    """

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), NUM_THREADS + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (NUM_THREADS, ranges))
    sys.stdout.flush()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, out_dir, filenames, num_shards, im_dim)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    for process in threads:
        process.join()

    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def find_image_files(data_dir):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir(str): Path to the root directory of images.
        Assumes that the image data set resides in JPEG files either
        located in data_dir or in the following directory structure.
          data_dir/dog/another-image.JPEG
          data_dir/dog/my-image.jpg
        where 'dog' is the label associated with these images.

    Returns:
      filenames: list of strings; each string is a path to an image file.
    """

    print('Determining list of input files and labels from %s.' % data_dir)

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of labels (from subfolders)
    unique_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not unique_labels:
        jpeg_file_pattern = '%s*.JPEG' % (data_dir)
        matching_files = glob.glob(jpeg_file_pattern)
        filenames.extend(matching_files)
    else:
        for text in unique_labels:
            jpeg_file_pattern = '%s%s/*' % (data_dir, text)
            matching_files = glob.glob(jpeg_file_pattern)
            filenames.extend(matching_files)
            if not label_index % 100:
                print('Finished finding files in %d of %d classes.' % (
                    label_index, len(filenames)))
            label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames


def process_dataset(name, src_dir, out_dir, num_shards, im_dim=(256, 256)):
    """Process a complete data set and save it as HDF5.

    Args:
      name: string, unique identifier specifying the data set.
      src_dir: string, root path to the data set.
      out_dir: Directory for storing results
      num_shards: integer number of shards for this data set.
      im_dim: Tuple (height, width) defining the size of the images
    """
    filenames = find_image_files(src_dir)
    process_image_files(name, out_dir, filenames, num_shards, im_dim)