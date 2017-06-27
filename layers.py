import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.python.ops import math_ops


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def up_conv2d(net, num_outputs, scope, factor=2, resize_fun=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, (factor * in_shape[1], factor * in_shape[2]), resize_fun)
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


def sample(mu, log_var):
    noise_shape = mu.get_shape().as_list()
    noise = tf.random_normal(shape=noise_shape)
    samples = math_ops.add(math_ops.mul(math_ops.exp(log_var / 2.0), noise), mu)
    return samples


def ordered_merge(a, b, order):
    return tf.select(tf.python.math_ops.greater(order, 0), merge(a, b), merge(b, a))


def random_select(a, b, p, batch_size):
    rand_vec = tf.random_uniform((batch_size,), minval=0.0, maxval=1.0)
    return tf.select(tf.python.math_ops.greater(rand_vec, p), a, b)


def merge(a, b, dim=3):
    return tf.concat(concat_dim=dim, values=[a, b])


def swap_merge(a, b):
    a1, a2 = tf.split(0, 2, a)
    b1, b2 = tf.split(0, 2, b)
    m1 = merge(a1, b1)
    m2 = merge(b2, a2)
    return merge(m1, m2, dim=0)


def spatial_shuffle(net, p):
    in_shape = net.get_shape().as_list()
    net = tf.transpose(net, [1, 2, 0, 3])
    net = tf.reshape(net, shape=[-1, in_shape[0], in_shape[3]])
    net_shuffled = tf.random_shuffle(net)
    net = random_select(net, net_shuffled, p, net.get_shape().as_list()[0])
    net = tf.reshape(net, [in_shape[1], in_shape[2], in_shape[0], in_shape[3]])
    net = tf.transpose(net, [2, 0, 1, 3])
    return net


def feature_dropout(net, p):
    input_shape = net.get_shape().as_list()
    noise_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
    return tf.nn.dropout(net, p, noise_shape=noise_shape, name='feature_dropout')


def spatial_dropout(net, p):
    input_shape = net.get_shape().as_list()
    noise_shape = (input_shape[0], 1, 1, input_shape[3])
    return tf.nn.dropout(net, p, noise_shape=noise_shape, name='spatial_dropout')


def conv_group(net, num_out, kernel_size, scope):
    input_groups = tf.split(split_dim=3, num_split=2, value=net)
    output_groups = [slim.conv2d(j, num_out/2, kernel_size=kernel_size, scope='{}_{}'.format(scope, idx))
                     for (idx, j) in enumerate(input_groups)]
    return tf.concat(concat_dim=3, values=output_groups)


def res_block(inputs, depth, depth_bottleneck, scope=None):
    with slim.variable_scope.variable_scope(scope):
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        shortcut = inputs
        residual = slim.conv2d(preact, depth_bottleneck, kernel_size=[1, 1], stride=1, scope='conv1')
        residual = slim.conv2d(residual, depth_bottleneck, kernel_size=[3, 3], stride=1, scope='conv2')
        residual = slim.conv2d(residual, depth, kernel_size=[1, 1], stride=1, normalizer_fn=None, activation_fn=None,
                               scope='conv3')
        output = shortcut + residual
        return output


def maxpool2d_valid(inputs, kernel_size, stride, scope=None):
    pad_total = kernel_size[0] - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.max_pool2d(inputs, kernel_size, stride=stride, padding='VALID', scope=scope)