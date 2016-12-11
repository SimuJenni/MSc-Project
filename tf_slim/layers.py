import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import math_ops


def bottleneck_res_layer(inputs, depth, depth_bottleneck, noise_channels, scope, reuse=None):
    """Bottleneck residual unit variant with BN before convolutions.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, reuse=reuse):
        shortcut = inputs
        residual = add_noise_plane(inputs, noise_channels)
        residual = slim.conv2d(residual, depth_bottleneck, kernel_size=(1, 1), stride=1, scope='conv1')
        residual = slim.conv2d(residual, depth_bottleneck, scope='conv2', stride=1)
        residual = slim.conv2d(residual, depth, kernel_size=(1, 1), stride=1, activation_fn=None, scope='conv3')
        output = tf.nn.relu(shortcut + residual)
        return output


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def up_conv2d(net, num_outputs, scope, factor=2):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, factor * in_shape[1], factor * in_shape[2],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    net = slim.conv2d(net, num_outputs=num_outputs, scope=scope, stride=1)
    return net


def add_noise_plane(net, noise_channels, training=True):
    noise_shape = net.get_shape().as_list()
    noise_shape[-1] = noise_channels
    noise_planes = tf.random_normal(shape=noise_shape)
    biases = tf.Variable(tf.constant(0.0, shape=[noise_channels], dtype=tf.float32), trainable=True, name='noise_mu')
    if training:
        slim.add_model_variable(biases)
    noise_planes = tf.nn.bias_add(noise_planes, biases)
    return tf.concat(3, [net, noise_planes], name='add_noise_{}'.format(noise_channels))


def sample(mu, log_sigma):
    noise_shape = mu.get_shape().as_list()
    noise = tf.random_normal(shape=noise_shape)
    samples = math_ops.add(math_ops.mul(math_ops.exp(log_sigma), noise), mu)
    return samples


def ordered_merge(a, b, order):
    return tf.select(tf.python.math_ops.greater(order, 0), merge(a, b), merge(b, a))


def random_select(a, b, p, batch_size):
    rand_vec = tf.random_uniform((batch_size,), minval=0.0, maxval=1.0)
    return tf.select(tf.python.math_ops.greater(rand_vec, p), a, b)


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
