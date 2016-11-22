import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_slim.layers import lrelu, up_conv2d, add_noise_plane, merge, spatial_dropout, feature_dropout

F_DIMS = [64, 96, 128, 256, 512, 1024, 2048]
NOISE_CHANNELS = [2, 4, 8, 16, 32, 64, 100]


class AEGAN3:
    def __init__(self, num_layers, batch_size):
        self.name = 'AEGANv3'
        self.num_layers = num_layers
        self.batch_size = batch_size

    def net(self, img, cartoon, edges, reuse=None):
        gen_in = merge(cartoon, edges)
        gen_enc = generator(gen_in, num_layers=self.num_layers, reuse=reuse)
        enc_im = encoder(img, num_layers=self.num_layers, reuse=reuse)
        dec_im = decoder(enc_im, num_layers=self.num_layers, reuse=reuse)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True)
        disc_in = merge(merge(merge(img, merge(dec_im, dec_gen)),
                              merge(dec_gen, merge(img, dec_im)), dim=0),
                        merge(dec_im, merge(dec_gen, img)), dim=0)
        disc_in += tf.random_normal(shape=tf.shape(disc_in),
                                    stddev=1.0 * tf.pow(0.975, tf.to_float(self.batch_size * slim.get_global_step() / 1000)))
        disc_out = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse)
        return dec_im, dec_gen, disc_out, enc_im, gen_enc

    def disc_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)

    def ea_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)

    def gen_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)


class AEGAN2:
    def __init__(self, num_layers, batch_size):
        self.name = 'AEGAN2_test'
        self.num_layers = num_layers
        self.batch_size = batch_size

    def net(self, img, cartoon, edges, labels=None, reuse=None):
        gen_in = merge(cartoon, edges)
        gen_enc = generator(gen_in, num_layers=self.num_layers, reuse=reuse)
        enc_im = encoder(img, num_layers=self.num_layers, reuse=reuse)
        dec_im = decoder(enc_im, num_layers=self.num_layers, reuse=reuse)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True)
        disc_in = merge(merge(dec_gen, cartoon), merge(dec_im, cartoon), dim=0)
        disc_in += tf.random_normal(shape=tf.shape(disc_in),
                                    stddev=1.0 * tf.pow(0.95,
                                                        tf.to_float(self.batch_size * slim.get_global_step() / 1000)))
        disc_out = discriminator(disc_in, num_layers=self.num_layers, batch_size=self.batch_size, reuse=reuse)
        return dec_im, dec_gen, disc_out, enc_im, gen_enc

    @staticmethod
    def disc_labels(batch_size):
        return tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(batch_size,), dtype=tf.int32),
                                                           tf.ones(shape=(batch_size,), dtype=tf.int32)]))


class AEGAN:
    def __init__(self, num_layers, batch_size):
        self.name = 'AEGAN'
        self.num_layers = num_layers
        self.batch_size = batch_size

    def net(self, img, cartoon, edges, labels=None):
        gen_in = merge(cartoon, edges)
        gen_enc = generator(gen_in, num_layers=self.num_layers)
        enc_im = encoder(img, num_layers=self.num_layers)
        disc_in = merge(enc_im, gen_enc, dim=0)
        disc_out = discriminator_ae(disc_in)
        dec_im = decoder(enc_im, num_layers=self.num_layers)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True)
        return dec_im, dec_gen, disc_out, enc_im, gen_enc

    @staticmethod
    def disc_labels(batch_size):
        return tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(batch_size,), dtype=tf.int32),
                                                           tf.ones(shape=(batch_size,), dtype=tf.int32)]))


def generator(inputs, num_layers=5, reuse=None):
    f_dims = F_DIMS
    with tf.variable_scope('generator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME')):
            net = add_noise_plane(inputs, NOISE_CHANNELS[0])
            net = slim.conv2d(net, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[l])
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))

            net = add_noise_plane(net, NOISE_CHANNELS[num_layers])
            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)

            return net


def encoder(inputs, num_layers=5, reuse=None):
    f_dims = F_DIMS
    with tf.variable_scope('encoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME')):
            net = slim.conv2d(inputs, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))

            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)
            return net


def decoder(inputs, num_layers=5, reuse=None):
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


def discriminator_ae(inputs, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu)):
            # Fully connected layers
            inputs += tf.random_normal(shape=tf.shape(inputs),
                                       stddev=5.0 * tf.pow(0.95, tf.to_float(slim.get_global_step() / 250)))
            # inputs = spatial_dropout(inputs, 0.5)
            inputs = feature_dropout(inputs, 0.5 * tf.pow(0.95, tf.to_float(slim.get_global_step() / 250)))
            net = slim.flatten(inputs)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net


def discriminator(inputs, num_layers=5, batch_size=128, reuse=None):
    f_dims = F_DIMS
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='VALID')):
            net = slim.conv2d(inputs, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))
            # Fully connected layers
            # net = feature_dropout(net, 0.5*tf.pow(0.95, tf.to_float(slim.get_global_step()/1000)))
            net = spatial_dropout(net, 0.5 * tf.pow(0.975, tf.to_float(batch_size * slim.get_global_step() / 1000)))
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net


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


def classifier(inputs, model, num_classes):
    net = model(inputs)
    batch_norm_params = {'decay': 0.9997, 'epsilon': 0.001}
    with tf.variable_scope('fully_connected'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net)
            net = slim.fully_connected(net, 4096, scope='fc2')
            net = slim.dropout(net)
            net = slim.fully_connected(net, num_classes, scope='fc3',
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)
    return net
