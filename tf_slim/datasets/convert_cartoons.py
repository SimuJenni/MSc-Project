from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from scipy import misc

from cartooning import auto_canny
import numpy as np
import tensorflow as tf

from tf_slim.datasets import dataset_utils


def _to_tfrecord(source_dir, tfrecord_writer):

    img_list = dir(source_dir)
    num_images = len(img_list)

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()

        with tf.Session('') as sess:
            for j in range(num_images):
                img_path = img_list[j]

                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (img_path, j + 1, num_images))
                sys.stdout.flush()

                # Get image, edge-map and cartooned image
                cartoon = misc.imread(img_path)
                im_shape = np.shape(cartoon)
                edges = auto_canny(cartoon)[:, :, None]

                # Encode the images
                cartoon_str = coder.encode_jpeg(cartoon)
                edge_str = coder.encode_jpeg(edges)

                # Buil example
                example = dataset_utils.cartooned_example(cartoon_str, cartoon_str, edge_str, 'jpg', im_shape[0],
                                                          im_shape[1], 0)
                tfrecord_writer.write(example.SerializeToString())


def run(target_dir, source_dir):
    """Runs the conversion operation.
    Args:
      target_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(target_dir):
        tf.gfile.MakeDirs(target_dir)

    # Process the test data:
    with tf.python_io.TFRecordWriter(target_dir) as tfrecord_writer:
        _to_tfrecord(source_dir, tfrecord_writer)

    print('\nFinished converting the cartoon dataset!')
