import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_slim.layers import lrelu, up_conv2d, add_noise_plane, merge, spatial_dropout, feature_dropout

F_DIMS = [64, 96, 128, 256, 512, 1024, 2048]
NOISE_CHANNELS = [1, 8, 16, 32, 64, 128, 256]


class AEGAN4:
    def __init__(self, num_layers, batch_size):
        self.name = 'AEGANv4_dtransp_0.05lsmooth'
        self.num_layers = num_layers
        self.batch_size = batch_size

    def net(self, img, cartoon, edges, reuse=None, training=True):
        gen_in = merge(cartoon, edges)
        gen_enc, _ = generator(gen_in, num_layers=self.num_layers, reuse=reuse, training=training)
        enc_im, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_im = decoder(enc_im, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True, training=training)
        disc_in = merge(merge(merge(img, merge(dec_im, dec_gen)),
                              merge(dec_gen, merge(img, dec_im)), dim=0),
                        merge(dec_im, merge(dec_gen, img)), dim=0)
        disc_in += tf.random_normal(shape=tf.shape(disc_in),
                                    stddev=noise_amount(10000/self.batch_size))
        disc_out, _ = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse, num_out=3,
                                    batch_size=self.batch_size, training=training,
                                    noise_level=noise_amount(5000/self.batch_size))
        return dec_im, dec_gen, disc_out, [enc_im], [gen_enc]

    def disc_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)

    def ae_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)

    def gen_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[2 * tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 3)

    def classifier(self, img, edge, toon, num_classes):
        disc_in = merge(img, merge(img, img))
        _, model = discriminator(disc_in, num_layers=self.num_layers, reuse=False, num_out=num_classes,
                                 batch_size=self.batch_size, training=True)
        model = classifier(model, num_classes)
        return model


class AEGAN2:
    def __init__(self, num_layers, batch_size):
        self.name = 'AEGANv2'
        self.num_layers = num_layers
        self.batch_size = batch_size

    def net(self, img, cartoon, edges, reuse=None, training=True):
        gen_in = merge(cartoon, edges)
        gen_enc, _ = generator(gen_in, num_layers=self.num_layers, reuse=reuse, p=0.75, training=training)
        enc_im, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, p=0.75, training=training)
        dec_im = decoder(enc_im, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True, training=training)
        disc_in = merge(merge(dec_gen, img), merge(dec_im, img), dim=0)
        disc_in += tf.random_normal(shape=tf.shape(disc_in),
                                    stddev=tf.pow(0.975, tf.to_float(self.batch_size * slim.get_global_step() / 1000)))
        disc_out, _ = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse, batch_size=self.batch_size,
                                    training=training)
        return dec_im, dec_gen, disc_out, [enc_im], [gen_enc]

    def disc_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)

    def ae_labels(self):
        return self.disc_labels()

    def gen_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)


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

    def disc_labels(self):
        return tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                           tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))


def generator(inputs, num_layers=5, reuse=None, p=1.0, training=True):
    f_dims = F_DIMS
    with tf.variable_scope('generator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = add_noise_plane(inputs, NOISE_CHANNELS[0])
            net = slim.conv2d(net, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)
            layers = []
            for l in range(1, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[l])
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))
                layers.append(net)

            net = add_noise_plane(net, NOISE_CHANNELS[num_layers])
            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)
            net = spatial_dropout(net, p)

            return net, layers


def encoder(inputs, num_layers=5, reuse=None, p=1.0, training=True):
    f_dims = F_DIMS
    with tf.variable_scope('encoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = slim.conv2d(inputs, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)
            layers = []
            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))
                layers.append(net)

            net = slim.conv2d(net, num_outputs=f_dims[num_layers], scope='conv_{}'.format(num_layers + 1), stride=1)
            net = spatial_dropout(net, p)
            return net, layers


def decoder(inputs, num_layers=5, reuse=None, layers=None, scope='decoder', training=True):
    f_dims = F_DIMS
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = slim.conv2d(inputs, f_dims[num_layers - 1], stride=1, scope='deconv_0')

            for l in range(1, num_layers):
                if layers:
                    net = merge(net, layers[-l])
                # net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))
                net = slim.conv2d_transpose(net, num_outputs=f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))


            net = slim.conv2d(net, num_outputs=3, scope='upconv_{}'.format(num_layers), stride=1,
                              activation_fn=tf.nn.tanh, padding='SAME')
            return net


def discriminator_ae(inputs, reuse=None, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, training=training)):
            # Fully connected layers
            inputs = feature_dropout(inputs, 1.0 - 0.5 * tf.pow(0.95, tf.to_float(slim.get_global_step() / 250)))
            net = slim.flatten(inputs)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net


def discriminator(inputs, noise_level, num_layers=5, reuse=None, num_out=2, batch_size=128, training=True):
    f_dims = F_DIMS
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='VALID', training=training)):
            net = slim.conv2d(inputs, kernel_size=(3, 3), num_outputs=f_dims[0], scope='conv_0', stride=1)

            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}'.format(l + 1))

            encoded = net
            # Fully connected layers
            net = spatial_dropout(net, 0.5)
            net = feature_dropout(net, 1.0 - 0.75 * noise_level)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, 2048)
            net = slim.dropout(net, 0.5)
            net = slim.fully_connected(net, num_out,
                                       activation_fn=None,
                                       normalizer_fn=None)
            return net, encoded


def toon_net_argscope(activation=tf.nn.relu, kernel_size=(4, 4), padding='SAME', training=True):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.999,
        'epsilon': 0.001,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            stride=2,
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


def noise_amount(decay_steps, decay_rate=0.9):
    return tf.train.exponential_decay(1.0, slim.get_global_step(), int(decay_steps), decay_rate, name='noise_rate')


def classifier(inputs, num_classes):
    batch_norm_params = {'decay': 0.99, 'epsilon': 0.001}
    with tf.variable_scope('fully_connected'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            net = slim.flatten(inputs)
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net)
            net = slim.fully_connected(net, 4096, scope='fc2')
            net = slim.dropout(net)
            net = slim.fully_connected(net, num_classes, scope='fc3',
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)
    return net
