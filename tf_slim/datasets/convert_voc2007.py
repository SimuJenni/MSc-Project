from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import xml.etree.ElementTree as ET
from scipy import misc

import cv2
import numpy as np
import tensorflow as tf
from constants import VOC2007_TF_DATADIR, VOC2007_SRC_DIR

from tf_slim.datasets import dataset_utils


# The names of the classes.
_CLASS_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

_CLASS_NUM = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
              'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
              'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


def _parse_xml(xml_file, data_path):
    """parse xml_file
    Args:
      xml_file: the input xml file path
    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_path = ''
    label = np.zeros(20, dtype=np.int64)

    for item in root:
        if item.tag == 'filename':
            image_path = os.path.join(data_path, 'VOC2007/JPEGImages', item.text)
        elif item.tag == 'object':
            obj_name = item[0].text
            obj_num = _CLASS_NUM[obj_name]
            label[obj_num] = 1

    return image_path, label


def _to_tfrecord(image_ids_file, tfrecord_writer, source_dir, max_im_dim=192):
    with open(image_ids_file) as f:
        img_ids = f.readlines()

    num_images = len(img_ids)

    with tf.Graph().as_default():
        coder = dataset_utils.ImageCoder()

        with tf.Session('') as sess:
            for j in range(num_images):
                xml_path = os.path.join(source_dir, 'VOC2007/Annotations', '{}.xml'.format(img_ids[j].strip('\n')))

                img_path, label = _parse_xml(xml_path, source_dir)

                sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (img_path, j + 1, num_images))
                sys.stdout.flush()

                # Get image, edge-map and cartooned image
                img = misc.imread(img_path)

                # Resize the image
                h = np.size(img, 0)
                w = np.size(img, 1)
                if w > h:
                    pic = img[0:h, int(round(w / 2 - h / 2)):int(round(w / 2 - h / 2) + h), :]
                    image = cv2.resize(pic, (max_im_dim, max_im_dim * w // h), interpolation=cv2.INTER_CUBIC)
                else:
                    pic = img[int(round(h / 2 - w / 2)):int(round(h / 2 - w / 2) + w), 0:w, :]
                    image = cv2.resize(pic, (max_im_dim * h // w, max_im_dim), interpolation=cv2.INTER_CUBIC)

                # Encode the images
                image_str = coder.encode_jpeg(image)

                # Buil example
                example = dataset_utils.image_to_tfexample(image_str, 'jpg', max_im_dim, max_im_dim, label.tolist())
                tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      split_name: The name of the train/test split.
    Returns:
      An absolute file path.
    """
    return '%s/voc2007_%s.tfrecord' % (dataset_dir, split_name)


def run(target_dir=VOC2007_TF_DATADIR, source_dir=VOC2007_SRC_DIR):
    """Runs the conversion operation.
    Args:
      target_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(target_dir):
        tf.gfile.MakeDirs(target_dir)

    train_filename = _get_output_filename(target_dir, 'train')
    val_filename = _get_output_filename(target_dir, 'val')
    trainval_filename = _get_output_filename(target_dir, 'trainval')
    testing_filename = _get_output_filename(target_dir, 'test')

    # First, process the trainval data:
    with tf.python_io.TFRecordWriter(trainval_filename) as tfrecord_writer:
        filename = os.path.join(source_dir, 'VOC2007/ImageSets/Main', 'trainval.txt')
        _to_tfrecord(filename, tfrecord_writer, source_dir)

    # Process the train data:
    with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
        filename = os.path.join(source_dir, 'VOC2007/ImageSets/Main', 'train.txt')
        _to_tfrecord(filename, tfrecord_writer, source_dir)

    # Process the val data:
    with tf.python_io.TFRecordWriter(val_filename) as tfrecord_writer:
        filename = os.path.join(source_dir, 'VOC2007/ImageSets/Main', 'val.txt')
        _to_tfrecord(filename, tfrecord_writer, source_dir)

    # Process the test data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = os.path.join(source_dir, 'VOC2007/ImageSets/Main', 'test.txt')
        _to_tfrecord(filename, tfrecord_writer, source_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    dataset_utils.write_label_file(labels_to_class_names, target_dir)

    print('\nFinished converting the VOC2007 dataset!')

if __name__ == '__main__':
    run()