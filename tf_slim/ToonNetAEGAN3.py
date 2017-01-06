import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops

from layers import lrelu, up_conv2d, sample, merge, add_noise_plane

DEFAULT_FILTER_DIMS = [64, 128, 256, 512, 512]
REPEATS = [1, 1, 2, 2, 2]
NOISE_CHANNELS = [1, 4, 8, 16, 32, 64, 128]


class VAEGAN:
    def __init__(self, num_layers, batch_size, data_size, num_epochs=100):
        """Initialises an AEGAN using the provided paramters.

        Args:
            num_layers: The number of convolutional down/upsampling layers to be used.
            batch_size: The batch-size used during training (used to generate training labels)
            data_size: Number of training images in the dataset
            num_epochs: Number of epochs used for training
        """
        self.name = 'AEGANv3_do'
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.data_size = data_size
        self.num_ep = num_epochs
        self.num_steps = num_epochs * data_size / batch_size

    def net(self, img, cartoon, edges, reuse=None, training=True):
        """Builds the AEGAN with the given inputs.

        Args:
            img: Placeholder for input images
            cartoon: Placeholder for cartooned images
            edges: Placeholder for edge-maps
            reuse: Whether to reuse already defined variables.
            training: Whether in train or test mode

        Returns:
            dec_im: The autoencoded image
            dec_gen: The reconstructed image from cartoon and edge inputs
            disc_out: The discriminator output
            enc_im: Encoding of the image
            gen_enc: Output of the generator
        """
        # Concatenate cartoon and edge for input to generator
        gen_in = merge(cartoon, edges)
        gen_dist, gen_mu, gen_logvar, _ = generator(gen_in, num_layers=self.num_layers, reuse=reuse, training=training)
        enc_dist, enc_mu, enc_logvar, _ = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        # Decode both encoded images and generator output using the same decoder
        dec_im = decoder(enc_dist, num_layers=self.num_layers, reuse=reuse, training=training)
        dec_gen = decoder(gen_dist, num_layers=self.num_layers, reuse=True, training=training)
        # Build input for discriminator (discriminator tries to guess order of real/fake)
        disc_in = merge(merge(dec_gen, dec_im), merge(dec_im, dec_gen), dim=0)
        disc_out, _, _ = discriminator(disc_in, num_layers=5, reuse=reuse, num_out=2, training=training)
        return dec_im, dec_gen, disc_out, enc_dist, gen_dist, enc_mu, gen_mu, enc_logvar, gen_logvar

    def disc_labels(self):
        """Generates labels for discriminator training (see discriminator input!)

        Returns:
            One-hot encoded labels
        """
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.ones(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)

    def gen_labels(self):
        """Generates labels for generator training (see discriminator input!). Exact opposite of disc_labels

        Returns:
            One-hot encoded labels
        """
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.ones(shape=(self.batch_size,), dtype=tf.int32),
                                                             tf.zeros(shape=(self.batch_size,), dtype=tf.int32)]))
        return slim.one_hot_encoding(labels, 2)

    def classifier(self, img, edge, num_classes, type='generator', reuse=None, training=True, fine_tune=True):
        """Builds a classifier on top either the encoder, generator or discriminator trained in the AEGAN.

        Args:
            img: Input image
            edge: Edge map
            toon: Cartooned image
            num_classes: Number of output classes
            type: Type of network to build on. One of {'generator', 'encoder', 'discriminator'}
            reuse: Whether to reuse already defined variables.
            training: Whether in train or test mode
            fine_tune: If False, builds classifier using encoder (used to compare to supervised training)

        Returns:
            Output logits from the classifier
        """
        activation = tf.nn.relu
        if not fine_tune:
            _, _, _, model = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        elif type == 'generator':
            gen_in = merge(img, edge)
            _, _, _, model = generator(gen_in, num_layers=self.num_layers, reuse=reuse, training=training)
        elif type == 'discriminator':
            disc_in = merge(img, img)
            _, model, _ = discriminator(disc_in, num_layers=self.num_layers, reuse=reuse, num_out=num_classes,
                                        training=training, train_fc=False, weights_reg=0.0001)
            activation = lrelu
        elif type == 'encoder':
            _, _, _, model = encoder(img, num_layers=self.num_layers, reuse=reuse, training=training)
        else:
            raise ('Wrong type!')
        model = classifier(model, num_classes, reuse=reuse, training=training, activation=activation)
        return model


def generator(net, num_layers=5, reuse=None, training=True):
    """Builds a generator with the given inputs. Noise is induced in all convolutional layers.

    Args:
        net: Input to the generator (i.e. cartooned image and/or edge-map)
        num_layers: Number of convolutional down-sampling layers
        reuse: Whether to reuse already defined variables
        training: Whether in train or test mode.

    Returns:
        Encoding of the input.
    """
    f_dims = DEFAULT_FILTER_DIMS
    num_layers = min(num_layers, 4)
    with tf.variable_scope('generator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = slim.conv2d(net, num_outputs=32, stride=1, scope='conv_0')
            for l in range(0, num_layers):
                net = add_noise_plane(net, NOISE_CHANNELS[l], training=training)
                net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l+1))

            encoded = net
            mu = slim.conv2d(net, num_outputs=f_dims[num_layers-1], scope='conv_mu', activation_fn=None,
                             normalizer_fn=None)
            log_var = slim.conv2d(net, num_outputs=f_dims[num_layers-1], scope='conv_sigma', activation_fn=None,
                                  normalizer_fn=None)
            if training:
                net = sample(mu, log_var)
            else:
                net = mu

            return net, mu, log_var, encoded


def encoder(net, num_layers=5, reuse=None, training=True):
    """Builds an encoder of the given inputs.

    Args:
        net: Input to the encoder (image)
        num_layers: Number of convolutional down-sampling layers
        reuse: Whether to reuse already defined variables
        training: Whether in train or test mode.

    Returns:
        Encoding of the input image.
    """
    f_dims = DEFAULT_FILTER_DIMS
    num_layers = min(num_layers, 4)
    with tf.variable_scope('encoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            net = slim.conv2d(net, num_outputs=32, stride=1, scope='conv_0')
            for l in range(0, num_layers):
                net = slim.conv2d(net, num_outputs=f_dims[l], stride=2, scope='conv_{}'.format(l+1))

            encoded = net
            mu = slim.conv2d(net, num_outputs=f_dims[num_layers-1], scope='conv_mu', activation_fn=None,
                             normalizer_fn=None)
            log_var = slim.conv2d(net, num_outputs=f_dims[num_layers-1], scope='conv_sigma', activation_fn=None,
                                  normalizer_fn=None)
            if training:
                net = sample(mu, log_var)
            else:
                net = mu
            return net, mu, log_var, encoded


def decoder(net, num_layers=5, reuse=None, training=True):
    """Builds a decoder on top of net.

    Args:
        net: Input to the decoder (output of encoder)
        num_layers: Number of convolutional up-sampling layers
        reuse: Whether to reuse already defined variables
        training: Whether in train or test mode.

    Returns:
        Decoded image with 3 channels.
    """
    f_dims = DEFAULT_FILTER_DIMS
    num_layers = min(num_layers, 4)
    with tf.variable_scope('decoder', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(padding='SAME', training=training)):
            for l in range(0, num_layers):
                net = up_conv2d(net, num_outputs=f_dims[num_layers - l - 1], scope='deconv_{}'.format(l))
            net = slim.conv2d(net, num_outputs=3, scope='deconv_{}'.format(num_layers),
                              activation_fn=tf.nn.tanh, normalizer_fn=None)
            return net


def discriminator(net, num_layers=5, reuse=None, num_out=2, training=True, train_fc=True, weights_reg=0.00001):
    """Builds a discriminator network on top of inputs.

    Args:
        net: Input to the discriminator
        num_layers: Number of convolutional down-sampling layers
        reuse: Whether to reuse already defined variables
        num_out: Number of outputs (2 by default)
        training: Whether in train or test mode.

    Returns:
        Resulting logits
    """
    f_dims = DEFAULT_FILTER_DIMS
    with tf.variable_scope('discriminator', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=lrelu, padding='SAME', training=training,
                                              weights_reg=weights_reg)):
            layers = []
            for l in range(0, num_layers):
                net = slim.repeat(net, REPEATS[l], slim.conv2d, num_outputs=f_dims[l], scope='conv_{}'.format(l+1))
                net = slim.max_pool2d(net, [2, 2], scope='pool_{}'.format(l + 1))
                layers.append(net)

            encoded = net
            # Fully connected layers
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_out,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer,
                                       trainable=train_fc)
            return net, encoded, layers


def classifier(net, num_classes, reuse=None, training=True, activation=tf.nn.relu):
    """Builds a classifier on top of inputs consisting of 3 fully connected layers.

    Args:
        inputs: The input layer to the classifier
        num_classes: Number of output classes
        reuse: Whether to reuse the weights (if already defined earlier)
        training: Whether in train or test mode
        activation: The default activation function for fully connected layers

    Returns:
        Resulting logits for all the classes
    """
    with tf.variable_scope('fully_connected', reuse=reuse):
        with slim.arg_scope(toon_net_argscope(activation=activation, training=training, weights_reg=0.0001)):
            net = slim.flatten(net)
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, 0.9, is_training=training)
            net = slim.fully_connected(net, 4096, scope='fc2')
            net = slim.dropout(net, 0.9, is_training=training)
            net = slim.fully_connected(net, num_classes, scope='fc3',
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       biases_initializer=tf.zeros_initializer)
    return net


def toon_net_argscope(activation=tf.nn.relu, kernel_size=(3, 3), padding='SAME', training=True, weights_reg=0.00001):
    """Defines default parameter values for all the layers used in ToonNet.

    Args:
        activation: The default activation function
        kernel_size: The default kernel size for convolution layers
        padding: The default border mode
        training: Whether in train or test mode

    Returns:
        An argscope
    """
    batch_norm_params = {
        'is_training': True,    #TODO:?
        'decay': 0.999,
        'epsilon': 0.001,
        'center': False,
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.convolution2d_transpose],
                        activation_fn=activation,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(weights_reg)):
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            kernel_size=kernel_size,
                            padding=padding):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.dropout], is_training=training) as arg_sc:
                    return arg_sc


def noise_amount(decay_steps):
    rate = math_ops.maximum(1.0 - math_ops.cast(slim.get_global_step(), tf.float32) / decay_steps, 0.0,
                            name='noise_rate')
    return rate
