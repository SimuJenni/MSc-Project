import tensorflow as tf
import tensorflow.contrib.slim as slim

F_DIMS = [64, 96, 128, 256, 512, 1024, 2048]
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
                net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 1], scope='upconv_{}'.format(l + 1))

            net = add_noise_plane(net, NOISE_CHANNELS[0])
            net = slim.conv2d(net, num_outputs=3, scope='upconv_{}'.format(num_layers), stride=1,
                              activation_fn=tf.nn.tanh)
            return net


def ToonDiscriminator(inputs, num_layers=5, is_training=True):
    f_dims = F_DIMS
    with tf.variable_scope('discriminator'):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='VALID', kernel_size=(3, 3))):
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


def ToonGenAE(inputs, num_layers=5):
    f_dims = F_DIMS
    with tf.variable_scope('generator'):
        with slim.arg_scope(toon_net_argscope(padding='SAME')):
            net = add_noise_plane(inputs, NOISE_CHANNELS[0])
            net = slim.conv2d(net, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[l])
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))

            net = add_noise_plane(net, NOISE_CHANNELS[num_layers + 1])
            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)

            return net


def ToonEncoder(inputs, num_layers=5):
    f_dims = F_DIMS
    with tf.variable_scope('encoder'):
        with slim.arg_scope(toon_net_argscope(padding='SAME')):
            net = slim.conv2d(inputs, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))

            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)
            return net


def ToonDecoder(inputs, num_layers=5, reuse=None):
    f_dims = F_DIMS
    with tf.variable_scope('decoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME')):
            net = slim.conv2d(inputs, f_dims[num_layers - 1], stride=1, scope='deconv_0')

            for l in range(1, num_layers):
                net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))
                # net = slim.convolution2d_transpose(net, f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))

            net = slim.conv2d(net, num_outputs=3, scope='upconv_{}'.format(num_layers), stride=1,
                              activation_fn=tf.nn.tanh, padding='SAME')
            return net


def ToonDiscAE(inputs):
    with tf.variable_scope('discriminator'):
        with slim.arg_scope(toon_net_argscope(activation=lrelu)):
            # Fully connected layers
            inputs += tf.random_normal(shape=tf.shape(inputs),
                                       stddev=5.0*tf.pow(0.99, tf.to_float(slim.get_global_step()/1000)))
            net = slim.flatten(inputs)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net


def GANAE(img, cartoon, edges, order, num_layers=5):
    gen_in = merge(cartoon, edges)
    gen_enc = ToonGenAE(gen_in, num_layers=num_layers)
    enc_im = ToonEncoder(img, num_layers=num_layers)
    disc_in = merge(enc_im, gen_enc, dim=0)
    disc_out = ToonDiscAE(disc_in)
    dec_im = ToonDecoder(enc_im, num_layers=num_layers)
    dec_gen = ToonDecoder(gen_enc, num_layers=num_layers, reuse=True)

    tf.histogram_summary('gen_enc', gen_enc)
    tf.histogram_summary('enc_im', enc_im)

    return dec_im, dec_gen, disc_out, enc_im, gen_enc


def merge(a, b, dim=3):
    return tf.concat(concat_dim=dim, values=[a, b])


def add_noise_plane(net, noise_channels):
    noise_shape = net.get_shape().as_list()
    noise_shape[-1] = noise_channels
    return tf.concat(3, [net, tf.random_normal(shape=noise_shape)], name='add_noise_{}'.format(noise_channels))


def up_conv2d(net, num_outputs, scope, factor=2):
    in_shape = net.get_shape().as_list()
    net = tf.image.resize_images(net, factor * in_shape[1], factor * in_shape[2],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    net = slim.conv2d(net, num_outputs=num_outputs, scope=scope, stride=1)
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
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            stride=2,
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
                return arg_sc
