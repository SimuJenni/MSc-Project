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
               dropout_keep_prob=0.5,
               scope='alexnet_v2'):
    with tf.variable_scope('alexnet_v2') as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                    net = slim.conv2d(inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
                    net = slim.max_pool2d(net, [5, 5], 2, scope='pool1')
                    net = slim.conv2d(net, 256, [5, 5], scope='conv2')
                    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                    net = slim.conv2d(net, 384, [3, 3], scope='conv3')
                    net = slim.conv2d(net, 384, [3, 3], scope='conv4')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 4096, scope='fc1', activation_fn=tf.nn.relu)
                    net = slim.dropout(net, keep_prob=dropout_keep_prob)
                    net = slim.fully_connected(net, 4096, scope='fc2', activation_fn=tf.nn.relu)
                    net = slim.dropout(net, keep_prob=dropout_keep_prob)
                    net = slim.fully_connected(net, num_classes, scope='fc3', activation_fn=None)
                    return net


alexnet_v2.default_image_size = 224
