from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import sys
import threading
from datetime import datetime

import numpy as np
from scipy import misc

from cartooning import cartoonify, auto_canny
from constants import NUM_THREADS
from keras.utils import imcrop_tosquare


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
    image_cartoon = cartoonify(image_data)
    image_edge = auto_canny(image_data)
    return image_cartoon, image_edge, image_data


def process_image_files_batch(thread_index, ranges, name, out_dir, filenames, im_dim):
    """Processes and saves list of images in 1 thread.

    Args:
      thread_index (int): Unique batch to run index is within [0, len(ranges)).
      ranges: List of pairs of integers specifying ranges of each batches to    analyze in parallel.
      name (str): Unique identifier specifying the data set
      out_dir: Diriectory for storing results
      filenames: List of strings; each string is a path to an image file
      im_dim: Tuple (height, width) defining the size of the images
    """

    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    files_in_batch = np.arange(ranges[thread_index][0], ranges[thread_index][1], dtype=int)
    for i in files_in_batch:
        filename = filenames[i]
        toon_im, edge_im, real_im = process_image(filename, im_dim)
        output_filename = '%s-%s' % (name, os.path.basename(filename))
        toon_path = os.path.join(out_dir, 'cartoon/', output_filename)
        edge_path = os.path.join(out_dir, 'edge/', output_filename)
        img_path = os.path.join(out_dir, 'image/', output_filename)
        misc.imsave(toon_path, toon_im, format='JPEG')
        misc.imsave(edge_path, edge_im, format='JPEG')
        misc.imsave(img_path, real_im, format='JPEG')
        if not counter % 500:
            print('[thread %d]: Processed %d of %d images in thread batch.' %
                  (thread_index, counter, num_files_in_thread))
            sys.stdout.flush()
        counter += 1


def process_image_files(name, out_dir, filenames, im_dim):
    """Process and save list of images.

    Args:
      name (str): Unique identifier specifying the data set
      out_dir: Diriectory for storing results
      filenames: List of strings; each string is a path to an image file
      im_dim: Tuple (height, width) defining the size of the images
    """
    if not os.path.exists(os.path.join(out_dir, 'X')):
        os.makedirs(os.path.join(out_dir, 'X'))
    if not os.path.exists(os.path.join(out_dir, 'Y')):
        os.makedirs(os.path.join(out_dir, 'Y'))

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
        args = (thread_index, ranges, name, out_dir, filenames, im_dim)
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

    filenames = []

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
    # random ordering of the images
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))
    return filenames


def process_dataset(name, src_dir, out_dir, im_dim=(256, 256)):
    """Process a complete data set and save it as JPEG files of given dimension.

    Args:
      name: string, unique identifier specifying the data set.
      src_dir: string, root path to the data set.
      out_dir: Directory for storing results
      num_shards: integer number of shards for this data set.
      im_dim: Tuple (height, width) defining the size of the images
    """
    filenames = find_image_files(src_dir)
    process_image_files(name, out_dir, filenames, im_dim)