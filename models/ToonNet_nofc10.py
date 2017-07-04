import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import lrelu, up_conv2d, conv_group, res_block_bottleneck, channel_dropout, pixel_dropout, spatial_shuffle, merge

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 512]
REPEATS = [1, 1, 2, 2, 2]


def toon_net_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, center=True,
                      w_reg=0.0001, fix_bn=False):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or eval mode
        center: Whether to use centering in batchnorm
        w_reg: Parameter for weight-decay

    Returns:
        An argscope
    """
    train_bn = training and not fix_bn
    batch_norm_params = {
        'is_training': train_bn,
        'decay': 0.99,
        'epsilon': 0.001,
        'center': center,
        'fused': train_bn
    }
    he = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG')
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(w_reg),
                        biases_regularizer=slim.l2_regularizer(w_reg),
                        weights_initializer=he):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


class ToonNet:
    def __init__(self, num_layers, batch_size, im_shape, tag='default', vgg_discriminator=False, fix_bn=False,
                 vanilla_alex=True):
        """Initialises a ToonNet using the provided parameters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
            vgg_discriminator: Whether to use VGG-A instead of AlexNet in the discriminator
        """
        self.name = 'ToonNet_{}'.format(tag)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.vgg_discriminator = vgg_discriminator
        if vgg_discriminator:
            self.discriminator = VGGA(fix_bn=fix_bn)
        else:
            if vanilla_alex:
                self.discriminator = AlexNet(fix_bn=fix_bn)
            else:
                self.discriminator = AlexNet_V2(fix_bn=fix_bn)

    def net(self, img, reuse=None, training=True):
        """Builds the full ToonNet architecture with the given inputs.

        Args:
            img: Placeholder for input images
            cartoon: Placeholder for cartooned images
            reuse: Whether to reuse already defined variables.
            training: Whether in train or eval mode

        Returns:
            dec_im: The autoencoded image
            dec_gen: The reconstructed image from cartoon and edge inputs
            disc_out: The discriminator output
            enc_im: Encoding of the image
            gen_enc: Output of the generator
        """
        # Concatenate cartoon and edge for input to generator
        enc_im = self.encoder(img, reuse=reuse, training=training)
        enc_channel_drop = self.generator(channel_dropout(enc_im, 0.3), tag='cdrop', reuse=reuse, training=training)
        enc_pixel_drop = self.generator(pixel_dropout(enc_im, 0.5), tag='pdrop', reuse=reuse, training=training)
        enc_shuffle1 = self.generator(spatial_shuffle(enc_im, 0.4), tag='shuffle', reuse=reuse, training=training)
        enc_shuffle2 = spatial_shuffle(enc_im, 0.2)
        enc_pool = self.generator(slim.max_pool2d(enc_im, [2, 2], stride=1, padding='VALID'), tag='pool', reuse=reuse,
                                  training=training)

        # Decode both encoded images and generator output using the same decoder
        dec_im = self.decoder(enc_im, reuse=reuse, training=training)
        dec_cdrop = self.decoder(enc_channel_drop, reuse=True, training=False)
        dec_pdrop = self.decoder(enc_pixel_drop, reuse=True, training=False)
        dec_shuffle1 = self.decoder(enc_shuffle1, reuse=True, training=False)
        dec_shuffle2 = self.decoder(enc_shuffle2, reuse=True, training=False)
        dec_pool = self.decoder(enc_pool, reuse=True, training=False)

        # Build input for discriminator (discriminator tries to guess order of real/fake)
        disc_in = tf.concat(0, [dec_im, dec_pdrop, dec_cdrop, dec_shuffle1, dec_shuffle2, dec_pool])
        disc_out, disc_enc = self.discriminator.discriminate(disc_in, reuse=reuse, training=training)

        return dec_im, dec_cdrop, dec_pdrop, dec_shuffle1, dec_shuffle2, dec_pool, disc_out

    def gen_labels(self):
        """Generates labels for discriminator training (see discriminator input!)

        Returns:
            One-hot encoded labels
        """
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(5 * self.batch_size,), dtype=tf.int32)]))
        weights = tf.concat(concat_dim=0,
                            values=[tf.ones(shape=(self.batch_size,)), 0.2*tf.ones(shape=(5*self.batch_size,))])
        return slim.one_hot_encoding(labels, 2), weights

    def disc_labels(self):
        """Generates labels for generator training (see discriminator input!). Exact opposite of disc_labels

        Returns:
            One-hot encoded labels
        """
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(5 * self.batch_size,), dtype=tf.int32)]))
        weights = tf.concat(concat_dim=0,
                            values=[tf.ones(shape=(self.batch_size,)), 0.2*tf.ones(shape=(5*self.batch_size,))])
        return slim.one_hot_encoding(labels, 2), weights

    def domain_labels(self):
        labels = tf.Variable(tf.concat(concat_dim=0,
                                       values=[i * tf.ones(shape=(self.batch_size,), dtype=tf.int32) for i in
                                               range(2)]))
        return slim.one_hot_encoding(labels, 2)

    def build_classifier(self, img, num_classes, reuse=None, training=True):
        """Builds a classifier on top either the encoder, generator or discriminator trained in the AEGAN.

        Args:
            img: Input image
            num_classes: Number of output classes
            reuse: Whether to reuse already defined variables.
            training: Whether in train or eval mode

        Returns:
            Output logits from the classifier
        """
        _, model = self.discriminator.discriminate(img, reuse=reuse, training=training, with_fc=False)
        model = self.discriminator.classify(model, num_classes, reuse=reuse, training=training)
        return model

    def generator(self, net, tag='default', reuse=None, training=True):
        """Builds a generator with the given inputs. Noise is induced in all convolutional layers.

        Args:
            net: Input to the generator (i.e. cartooned image and/or edge-map)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Encoding of the input.
        """
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope(tag, reuse=reuse):
                with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                    shortcut = net
                    residual = slim.conv2d(net, 512, kernel_size=[3, 3], stride=1, scope='conv1')
                    residual = slim.conv2d(residual, 512, kernel_size=[3, 3], stride=1, scope='conv2',
                                           normalizer_fn=None, activation_fn=None)
                    output = slim.batch_norm(shortcut + residual, activation_fn=tf.nn.relu, scope='out')
                    return output

    def encoder(self, net, reuse=None, training=True):
        """Builds an encoder of the given inputs.

        Args:
            net: Input to the encoder (image)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Encoding of the input image.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                for l in range(0, self.num_layers):
                    net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l + 1))

                return net

    def decoder(self, net, reuse=None, training=True):
        """Builds a decoder on top of net.

        Args:
            net: Input to the decoder (output of encoder)
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.

        Returns:
            Decoded image with 3 channels.
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
                for l in range(0, self.num_layers-1):
                    net = up_conv2d(net, num_outputs=f_dims[self.num_layers - l - 2], scope='deconv_{}'.format(l))
                net = tf.image.resize_images(net, (self.im_shape[0], self.im_shape[1]),
                                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(self.num_layers), stride=1,
                                  activation_fn=tf.nn.tanh, normalizer_fn=None)
                return net


class AlexNet_V2:
    def __init__(self, fc_activation=tf.nn.relu, fix_bn=False):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation

    def classify(self, net, num_classes, reuse=None, training=True, scope='fully_connected'):
        """Builds a classifier on top of inputs consisting of 3 fully connected layers.

        Args:
            net: The input layer to the classifier
            num_classes: Number of output classes
            reuse: Whether to reuse the weights (if already defined earlier)
            training: Whether in train or eval mode

        Returns:
            Resulting logits for all the classes
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=self.fc_activation, training=training,
                                                  fix_bn=self.fix_bn)):
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, num_classes, scope='fc3',
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer)
        return net

    def discriminate(self, net, reuse=None, training=True, with_fc=True, pad='VALID', fix_bn=False):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='SAME', training=training,
                                                  fix_bn=self.fix_bn or fix_bn)):
                net = slim.conv2d(net, 64, kernel_size=[11, 11], stride=4, padding=pad, scope='conv_1',
                                  normalizer_fn=None)
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1')
                net = slim.conv2d(net, 192, kernel_size=[5, 5], scope='conv_2')
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2')
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3')
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_4')
                net = slim.conv2d(net, 256, kernel_size=[3, 3], scope='conv_5')
                encoded = net

                if with_fc:
                    # Fully connected layers
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 4096, scope='fc1', trainable=with_fc)
                    net = slim.dropout(net, 0.5, is_training=training)
                    net = slim.fully_connected(net, 4096, scope='fc2', trainable=with_fc)
                    net = slim.dropout(net, 0.5, is_training=training)
                    net = slim.fully_connected(net, 2,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               biases_initializer=tf.zeros_initializer,
                                               trainable=with_fc)
                return net, encoded

    def domain_classifier(self, net, num_classes, reuse=None, training=True, scope='dom_class'):
        return None


class AlexNet:
    def __init__(self, fc_activation=lrelu, fix_bn=False):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation

    def classify(self, net, num_classes, reuse=None, training=True, scope='fully_connected'):
        """Builds a classifier on top of inputs consisting of 3 fully connected layers.

        Args:
            net: The input layer to the classifier
            num_classes: Number of output classes
            reuse: Whether to reuse the weights (if already defined earlier)
            training: Whether in train or eval mode

        Returns:
            Resulting logits for all the classes
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=self.fc_activation, training=training,
                                                  fix_bn=self.fix_bn)):
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, num_classes, scope='fc3',
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer)
        return net

    def discriminate(self, net, reuse=None, training=True, with_fc=True, pad='VALID', fix_bn=False):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=self.fc_activation, padding='SAME', training=training,
                                                  fix_bn=self.fix_bn or fix_bn)):
                net = slim.conv2d(net, 96, kernel_size=[11, 11], stride=4, padding=pad, scope='conv_1',
                                  normalizer_fn=None)
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_1', padding='SAME')
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                net = conv_group(net, 256, kernel_size=[5, 5], scope='conv_2')
                net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_2')
                net = tf.nn.lrn(net, depth_radius=2, alpha=0.00002, beta=0.75)
                encoded = net
                net = slim.conv2d(net, 384, kernel_size=[3, 3], scope='conv_3')
                net = conv_group(net, 384, kernel_size=[3, 3], scope='conv_4')
                net = conv_group(net, 256, kernel_size=[3, 3], scope='conv_5')

                if with_fc:
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool_5')

                    net = slim.conv2d(net, 4096, kernel_size=[5, 5], padding='VALID', scope='fc6')
                    net = slim.dropout(net, 0.5, is_training=training, scope='dropout6')
                    net = slim.conv2d(net, 4096, kernel_size=[1, 1], scope='fc7')
                    net = slim.dropout(net, 0.5, is_training=training, scope='dropout7')
                    net = slim.conv2d(net, 2, kernel_size=[1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer,
                                      scope='fc8')
                    net = slim.flatten(net)

                return net, encoded

    def domain_classifier(self, net, num_classes, reuse=None, training=True, scope='dom_class'):
        """Builds a classifier on top of inputs consisting of 3 fully connected layers.

        Args:
            net: The input layer to the classifier
            num_classes: Number of output classes
            reuse: Whether to reuse the weights (if already defined earlier)
            training: Whether in train or eval mode

        Returns:
            Resulting logits for all the classes
        """
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=self.fc_activation, training=training,
                                                  fix_bn=self.fix_bn)):
                net = slim.conv2d(net, 256, kernel_size=[3, 3], scope='conv_3')
                net = slim.conv2d(net, num_classes, kernel_size=[5, 5], stride=1, scope='conv_4', activation_fn=None,
                                  padding='VALID', normalizer_fn=None)
                net = slim.flatten(net)
                return net


class VGGA:
    def __init__(self, fc_activation=tf.nn.relu, fix_bn=False):
        self.fix_bn = fix_bn
        self.fc_activation = fc_activation

    def classify(self, net, num_classes, reuse=None, training=True):
        """Builds a classifier on top of inputs consisting of 3 fully connected layers.

        Args:
            net: The input layer to the classifier
            num_classes: Number of output classes
            reuse: Whether to reuse the weights (if already defined earlier)
            training: Whether in train or eval mode

        Returns:
            Resulting logits for all the classes
        """
        with tf.variable_scope('fully_connected', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=self.fc_activation, training=training,
                                                  fix_bn=self.fix_bn)):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net, 0.5, is_training=training)
                net = slim.fully_connected(net, num_classes, scope='fc3',
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer)
        return net

    def discriminate(self, net, reuse=None, training=True, with_fc=True):
        """Builds a discriminator network on top of inputs.

        Args:
            net: Input to the discriminator
            reuse: Whether to reuse already defined variables
            training: Whether in train or eval mode.
            with_fc: Whether to include fully connected layers (used during unsupervised training)

        Returns:
            Resulting logits
        """
        f_dims = DEFAULT_FILTER_DIMS
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='SAME', training=training,
                                                  fix_bn=self.fix_bn)):
                for l in range(0, 5):
                    if l == 0:
                        net = slim.conv2d(net, f_dims[l], scope='conv_1_1', normalizer_fn=None)
                        net = slim.conv2d(net, f_dims[l], scope='conv_1_2')
                    else:
                        net = slim.repeat(net, REPEATS[l], slim.conv2d, num_outputs=f_dims[l],
                                          scope='conv_{}'.format(l + 1))
                    net = slim.max_pool2d(net, [2, 2], scope='pool_{}'.format(l + 1))

                encoded = net
                # Fully connected layers
                net = slim.flatten(net)
                net = slim.fully_connected(net, 2,
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer,
                                           trainable=with_fc)
                return net, encoded

    def domain_classifier(self, net, num_classes, reuse=None, training=True, scope='dom_class'):
        return None
