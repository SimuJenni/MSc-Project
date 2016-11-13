"""Contains a model definition for AlexNet.
This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014
Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)
@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v2(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               use_batch_norm=False):
    """AlexNet version 2.
    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg
    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224. To use in fully
          convolutional mode, set spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        outputs. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
    Returns:
      the last op containing the log predictions and end_points dict.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    with tf.variable_scope('alexnet_v2') as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params
                            ):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1', padding='VALID')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2', padding='SAME')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2', padding='VALID')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3', padding='SAME')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4', padding='SAME')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5', padding='SAME')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5', padding='VALID')

            # Use conv2d instead of fully_connected layers.
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=trunc_normal(0.005),
                                biases_initializer=tf.constant_initializer(0.1)):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.fully_connected(net, 4096, scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.fully_connected(net, num_classes,
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer,
                                           scope='fc8')

            return net


def alexnet(inputs,
            num_classes=1000,
            is_training=True,
            dropout_keep_prob=0.5,
            use_batch_norm=False):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    with tf.variable_scope('alexnet_v2') as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                net = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [5, 5], 2, scope='pool1')
                net = slim.conv2d(net, 256, [5, 5], scope='conv2', padding='SAME')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = slim.conv2d(net, 384, [3, 3], scope='conv3', padding='SAME')
                net = slim.conv2d(net, 384, [3, 3], scope='conv4', padding='SAME')
                net = slim.conv2d(net, 256, [3, 3], scope='conv5', padding='SAME')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training)
                net = slim.fully_connected(net, num_classes, scope='fc3', activation_fn=None)
                return net


alexnet.default_image_size = 224
