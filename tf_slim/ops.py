import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    assigners = []
    shadow_variables = []

    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, is_train, convolutional=True, decay=0.99, epsilon=1e-5, scale_after_normalization=True,
                 name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.convolutional = convolutional
            self.is_train = is_train
            self.epsilon = epsilon
            self.ema = tf.train.ExponentialMovingAverage(decay=decay)
            self.scale_after_normalization = scale_after_normalization
            self.name=name

    def __call__(self, x):
        shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            depth = shape[-1]
            self.gamma = tf.get_variable("gamma", shape=[depth],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", shape=[depth],
                                initializer=tf.constant_initializer(0.))
            self.mean = tf.get_variable('mean', shape=[depth],
                                        initializer=tf.constant_initializer(0),
                                        trainable=False)
            self.variance = tf.get_variable('variance', shape=[depth],
                                        initializer=tf.constant_initializer(1),
                                        trainable=False)
            
            # Add to assigners if not already added previously.
            if not tf.get_variable_scope().reuse:
                batch_norm.assigners.append(self.ema.apply([self.mean, self.variance]))
                batch_norm.shadow_variables += [self.ema.average(self.mean), self.ema.average(self.variance)]

            if self.convolutional:
                x_unflattened = x
            else:
                x_unflattened = tf.reshape(x, [-1, 1, 1, depth])

            if self.is_train:
                if self.convolutional:
                    mean, variance = tf.nn.moments(x, [0, 1, 2])
                else:
                    mean, variance = tf.nn.moments(x, [0])

                assign_mean = self.mean.assign(mean)
                assign_variance = self.variance.assign(variance)
                with tf.control_dependencies([assign_mean, assign_variance]):
                    normed = tf.nn.batch_norm_with_global_normalization(
                        x_unflattened, mean, variance, self.beta, self.gamma, self.epsilon,
                        scale_after_normalization=self.scale_after_normalization)
            else:
                mean = self.ema.average(self.mean)
                variance = self.ema.average(self.variance)
                local_beta = tf.identity(self.beta)
                local_gamma = tf.identity(self.gamma)
                normed = tf.nn.batch_norm_with_global_normalization(
                      x_unflattened, mean, variance, local_beta, local_gamma,
                      self.epsilon, self.scale_after_normalization)
            if self.convolutional:
                return normed
            else:
                return tf.reshape(normed, [-1, depth])

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if not tf.get_variable_scope().reuse:
            tf.histogram_summary(w.name, w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        if not tf.get_variable_scope().reuse:
            tf.histogram_summary(w.name, w)
        return tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1])

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope='Linear', stddev=0.02):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        if not tf.get_variable_scope().reuse:
            tf.histogram_summary(matrix.name, matrix)
        return tf.matmul(input_, matrix)


def normalize_batch_of_images(batch_of_images):
    mean, var = tf.nn.moments(batch_of_images, [1,2], keep_dims=True)
    std = tf.sqrt(var)
    normed = (batch_of_images - mean) / std
    return normed