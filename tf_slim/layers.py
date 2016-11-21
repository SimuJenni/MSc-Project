import tensorflow as tf
from tensorflow.contrib import slim as slim


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def up_conv2d(net, num_outputs, scope, factor=2):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, factor * in_shape[1], factor * in_shape[2],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    net = slim.conv2d(net, num_outputs=num_outputs, scope=scope, stride=1)
    return net


def add_noise_plane(net, noise_channels):
    noise_shape = net.get_shape().as_list()
    noise_shape[-1] = noise_channels
    return tf.concat(3, [net, tf.random_normal(shape=noise_shape)], name='add_noise_{}'.format(noise_channels))


def ordered_merge(a, b, order):
    return tf.select(tf.python.math_ops.greater(order, 0), merge(a, b), merge(b, a))


def merge(a, b, dim=3):
    return tf.concat(concat_dim=dim, values=[a, b])


def feature_dropout(net, p):
    input_shape = net.get_shape().as_list()
    noise_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
    return tf.nn.dropout(net, p, noise_shape=noise_shape, name='feature_dropout')


def spatial_dropout(net, p):
    input_shape = net.get_shape().as_list()
    noise_shape = (input_shape[0], 1, 1, input_shape[3])
    return tf.nn.dropout(net, p, noise_shape=noise_shape, name='spatial_dropout')