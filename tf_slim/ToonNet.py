import tensorflow as tf
import tensorflow.contrib.slim as slim

from tf_slim.layers import lrelu, up_conv2d, add_noise_plane, merge, spatial_dropout, random_select

F_DIMS = [64, 96, 128, 256, 512, 1024, 2048]
NOISE_CHANNELS = [1, 4, 8, 16, 32, 64, 128]


class AEGAN:
    def __init__(self, num_layers, batch_size, data_size, num_epochs):
        #self.name = 'AEGANv2_normal' #TODO: before
        self.name = 'AEGANv2'
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.data_size = data_size
        self.num_ep = num_epochs
        self.num_steps = num_epochs*data_size/batch_size

    def net(self, img, cartoon, edges, reuse=None, training=True):
        gen_in = merge(cartoon, edges)
        gen_enc, _ = generator(gen_in, num_layers=self.num_layers, reuse=reuse, training=training)
        enc_im, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_im = decoder(enc_im, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_gen = decoder(gen_enc, num_layers=self.num_layers, reuse=True, training=training)
        disc_in = merge(merge(dec_gen, dec_im), merge(dec_im, dec_gen), dim=0)
        disc_out, _, _ = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse, num_out=2, training=training)
        return dec_im, dec_gen, disc_out, [enc_im], [gen_enc]

    def disc_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)

    def gen_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)

    def classifier(self, img, edge, toon, num_classes, type='generator', reuse=None, training=True, fine_tune=True):
        activation = tf.nn.relu
        if not fine_tune:
            model, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        elif type == 'generator':
            gen_in = merge(img, edge)
            model, _ = generator(gen_in, num_layers=self.num_layers, reuse=reuse, training=training)
        elif type == 'discriminator':
            disc_in = merge(img, img)
            _, model, _ = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse, num_out=num_classes,
                                        training=training)
            activation = lrelu
        elif type == 'encoder':
            model, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        else:
            raise ('Wrong type!')
        model = classifier(model, num_classes, reuse=reuse, training=training, activation=activation)
        return model


def generator(inputs, num_layers=5, reuse=None, training=True):
    f_dims = F_DIMS
    with tf.variable_scope('generator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = add_noise_plane(inputs, NOISE_CHANNELS[0], training=training)
            net = slim.conv2d(net, num_outputs=f_dims[0], scope='conv_1', stride=1)
            layers = []
            for l in range(1, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[l], training=training)
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_1'.format(l + 1))
                net = add_noise_plane(net, NOISE_CHANNELS[l], training=training)
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_2'.format(l + 1), stride=1)
                layers.append(net)

            return net, layers


def encoder(inputs, num_layers=5, reuse=None, training=True):
    f_dims = F_DIMS
    with tf.variable_scope('encoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = slim.conv2d(inputs, num_outputs=f_dims[0], scope='conv_1', stride=1)
            layers = []
            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_1'.format(l + 1))
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_2'.format(l + 1), stride=1)
                layers.append(net)

            return net, layers


def decoder(net, num_layers=5, reuse=None, layers=None, scope='decoder', training=True):
    f_dims = F_DIMS
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            # net = spatial_dropout(net, 0.9) #TODO: Try without
            for l in range(1, num_layers):
                if layers:
                    net = merge(net, layers[-l])
                net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))
            net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(num_layers), stride=1,
                              activation_fn=tf.nn.tanh, normalizer_fn=None)
            return net


def discriminator(inputs, num_layers=5, reuse=None, num_out=2, training=True, noise_level=0.0):
    f_dims = F_DIMS
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='SAME', training=training)):
            net = slim.conv2d(inputs, num_outputs=f_dims[0], scope='conv_1', stride=1)
            layers = []
            for l in range(1, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_1'.format(l + 1))
                net = slim.conv2d(net, num_outputs=f_dims[l], scope='conv_{}_2'.format(l + 1), stride=1)
                layers.append(net)

            # net = spatial_dropout(net, 1.0 - noise_level) #TODO: before in
            encoded = net
            # Fully connected layers
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048)   # TODO:Before 4096
            net = slim.dropout(net, 0.9)
            net = slim.fully_connected(net, 2048)   # TODO:Before 4096
            net = slim.dropout(net, 0.9)
            net = slim.fully_connected(net, num_out, activation_fn=None, normalizer_fn=None)
            return net, encoded, layers


def toon_net_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.999,
        'epsilon': 0.001,
        'center': False,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00001)):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            stride=2,
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


def noise_amount(decay_steps):
    rate = tf.maximum(1.0 - tf.cast(slim.get_global_step(), tf.float32) / decay_steps, 0.0, name='noise_rate')
    return rate


def classifier(inputs, num_classes, reuse=None, training=True, activation=tf.nn.relu):
    batch_norm_params = {
        'is_training': training,
        'decay': 0.999,
        'epsilon': 0.001,
    }
    with tf.variable_scope('fully_connected', reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=activation,
                            weights_regularizer=slim.l2_regularizer(0.00001),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            net = slim.flatten(inputs)
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, 0.9, is_training=training)
            net = slim.fully_connected(net, 4096, scope='fc2')
            net = slim.dropout(net, 0.9, is_training=training)
            net = slim.fully_connected(net, num_classes, scope='fc3',
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)
    return net
