import tensorflow as tf
import tensorflow.contrib.slim as slim

F_DIMS = [64, 128, 256, 512, 1024, 2048]
NOISE_CHANNELS = [2, 4, 8, 16, 32, 64, 100]


def ToonGenerator(inputs, num_layers=5):
    f_dims = F_DIMS
    with tf.variable_scope('generator'):
        with toon_net_argscope():
            net = slim.conv2d(inputs, num_outputs=f_dims[0], scope='conv_0', stride=2)
            for l in range(0, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1), stride=2)

            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)
            net = add_noise_plane(net, NOISE_CHANNELS[num_layers + 1])
            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='upconv_0', stride=1)

            for l in range(0, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[num_layers - l])
                net = up_conv2d(net, depth=f_dims[num_layers - l - 1], scope='upconv_{}'.format(l + 1))

            net = add_noise_plane(net, NOISE_CHANNELS[0])
            net = slim.conv2d(net, num_outputs=3, scope='upconv_{}'.format(num_layers), stride=1,
                              activation_fn=tf.nn.tanh)
            return net


def ToonDiscriminator(inputs, num_layers=5, is_training=True):
    f_dims = F_DIMS
    with tf.variable_scope('discriminator'):
        with toon_net_argscope(activation=lrelu, padding='VALID', kernel_size=(3, 3)):
            net = slim.conv2d(inputs, num_outputs=f_dims[0], scope='conv_0', stride=2)
            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l), stride=2)

            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers), stride=1)

            # Fully connected layers
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5, is_training=is_training)
            net = slim.fully_connected(net, 1,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net


def add_noise_plane(net, noise_channels):
    noise_shape = net.get_shape().as_list()
    noise_shape[-1] = noise_channels
    return tf.concat(3, [net, tf.random_normal(shape=noise_shape)], name='add_noise_{}'.format(noise_channels))


def up_conv2d(net, depth, scope, factor=2):
    in_shape = net.get_shape()
    net = tf.image.resize_images(net, factor * in_shape[1], factor * in_shape[2],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    net = slim.conv2d(net, num_outputs=depth, scope=scope, stride=1)
    return net


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def toon_net_argscope(activation=tf.nn.relu, kernel_size=(4, 4), padding='SAME'):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=activation,
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc
