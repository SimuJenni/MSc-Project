import os

import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, merge, Dense, UpSampling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from constants import MODEL_DIR

F_DIMS = [64, 96, 160, 256, 512]
BF_DIMS = [64, 128, 256, 512, 1024]


def ToonGenerator(in_layer, out_activation='tanh', num_res_layers=8, big_f=False, outter=True, bn_mode=0):
    """Constructs a fully convolutional residual auto-encoder network.
    The network has the follow architecture:

    Layer           Filters     Stride  Connected

    L1: Conv-layer  4x4x64      1       L25            =================================||
    L2: Conv-layer  3x3x96      2       L24                =========================||  ||
    L3: Conv-layer  3x3x128     2       L23                    =================||  ||  ||
    L4: Conv-layer  3x3x256     2       L22                        =========||  ||  ||  ||
    L5: Conv-layer  3x3x512     2                                     ===   ||  ||  ||  ||
                                                                      |_|   ||  ||  ||  ||
    L6:                                                               |_|   ||  ||  ||  ||
    .               1x1x64      1                                     |_|   ||  ||  ||  ||
    .   Res-Layers  3x3x64      1                                     |_|   O4  O3  O2  O1
    .               3x3x512     1                                     |_|   ||  ||  ||  ||
    L20:                                                              |_|   ||  ||  ||  ||
                                                                      |_|   ||  ||  ||  ||
    L21: UpConv     3x3x256     2                                     ===   ||  ||  ||  ||
    L22: UpConv     3x3x128     2       L4                         =========||  ||  ||  ||
    L23: UpConv     3x3x96      2       L3                     =================||  ||  ||
    L24: UpConv     3x3x64      2       L2                 =========================||  ||
    L25: UpConv     4x4x64      1       L1             =================================||

    Args:
        merge_mode: Mode for merging the outer connections
        batch_size: Number of images per batch
        out_activation: Type of activation for last layer ('relu', 'sigmoid', 'tanh', ...)
        num_res_layers: Number of residual layers in the middle

    Returns:
        (net, encoded): The resulting Keras model (net) and the encoding layer
    """
    if big_f:
        f_dims = BF_DIMS
    else:
        f_dims = F_DIMS

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_relu(in_layer, f_size=3, f_channels=32, stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same', bn_mode=bn_mode)
        if outter:
            l1 = outter_connection(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', bn_mode=bn_mode)
        if outter:
            l2 = outter_connection(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same', bn_mode=bn_mode)
        if outter:
            l3 = outter_connection(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same', bn_mode=bn_mode)
        if outter:
            l4 = outter_connection(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same', bn_mode=bn_mode)
        if outter:
            l5 = outter_connection(x, f_dims[4])

    encoded = x

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[1], bn_mode=bn_mode, lightweight=True)
    if outter:
        x = merge([x, l5], mode='sum')

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        if outter:
            x = merge([x, l4], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        if outter:
            x = merge([x, l3], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        if outter:
            x = merge([x, l2], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        if outter:
            x = merge([x, l1], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=32, stride=1, border='same')
        x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
        decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonDiscriminator(in_layer, num_res_layers=8, big_f=False):
    """Builds ConvNet used as discrimator between real-images and de-tooned images.
    The network has the follow architecture:

    Layer           Filters     Stride

    L1: Conv-layer  3x3x32      2                  =================================
    L2: Conv-layer  3x3x64      2                      =========================
    L3: Conv-layer  3x3x128     2                          =================
    L4: Conv-layer  3x3x256     2                              =========
    L5: Conv-layer  3x3x512     2                                 ===
    L6: Dense-Layer 1024                                          |X|
    L7: Dense-Layer 1                                             |X|

    Args:
        input_shape: Shape of the input (height, width, channels)

    Returns:
        The resulting Keras models
    """

    if big_f:
        f_dims = BF_DIMS
    else:
        f_dims = F_DIMS

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_lrelu(in_layer, f_size=3, f_channels=32, stride=1, border='same')
        l1 = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same')

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_lrelu(l1, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        l2 = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_lrelu(l2, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        l3 = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same')

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_lrelu(l3, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        l4 = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same')

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_lrelu(l4, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        l5 = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same')

    layer_activations = [l1, l2, l3, l4, l5]
    x = l5

    # Res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck_lrelu(x, f_dims[4], f_dims[1], lightweight=True)

    # Fully connected layer
    x = GlobalAveragePooling2D()(x) # Maybe not a good idea?
    #x = Flatten()(x)
    x = Dense(2048, init='he_normal')(x)
    x = lrelu(x)
    x = BatchNormalization(axis=1)(x)
    layer_activations.append(x)
    x = Dense(1, init='he_normal')(x)
    x = Activation('sigmoid')(x)

    return x, layer_activations


def conv_relu_bn(layer_in, f_size, f_channels, stride, border='valid', activation='relu', bn_mode=0):
    """Wrapper for first few down-convolution layers including batchnorm and Relu

    Args:
        layer_in: Input to this layer
        f_size: Size of convolution filters
        f_channels: Number of channels of the output
        stride: Used stride (typically =2)
        border: 'valid' or 'same'

    Returns:
        Result of convolution followed by batchnorm and Relu
    """
    x = Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal')(layer_in)
    x = Activation(activation)(x)
    return BatchNormalization(axis=3, mode=bn_mode)(x)


def conv_lrelu_bn(layer_in, f_size, f_channels, stride, border='valid', bn_mode=0):
    """Wrapper for first few down-convolution layers including batchnorm and Relu

    Args:
        layer_in: Input to this layer
        f_size: Size of convolution filters
        f_channels: Number of channels of the output
        stride: Used stride (typically =2)
        border: 'valid' or 'same'

    Returns:
        Result of convolution followed by batchnorm and Relu
    """
    x = Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal')(layer_in)
    x = lrelu(x)
    return BatchNormalization(axis=3, mode=bn_mode)(x)


def conv_relu(layer_in, f_size, f_channels, stride, border='valid', activation='relu'):
    """Wrapper for first few down-convolution layers including batchnorm and Relu

    Args:
        layer_in: Input to this layer
        f_size: Size of convolution filters
        f_channels: Number of channels of the output
        stride: Used stride (typically =2)
        border: 'valid' or 'same'

    Returns:
        Result of convolution followed by batchnorm and Relu
    """
    x = Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal')(layer_in)
    return Activation(activation)(x)


def conv_lrelu(layer_in, f_size, f_channels, stride, border='valid'):
    """Wrapper for first few down-convolution layers including batchnorm and Relu

    Args:
        layer_in: Input to this layer
        f_size: Size of convolution filters
        f_channels: Number of channels of the output
        stride: Used stride (typically =2)
        border: 'valid' or 'same'

    Returns:
        Result of convolution followed by batchnorm and Relu
    """
    x = Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal')(layer_in)
    return lrelu(x)


def outter_connection(layer_in, f_channels, bn_mode=0):
    """Wrapper for 1x1 convolutions used on the outer layers.

    Args:
        layer_in: Input to this layer
        f_channels: Number of channels for the output (usually the same as input)

    Returns:
        Result of convolution followed by batchnorm and leakyRelu
    """
    l = Convolution2D(f_channels, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal')(layer_in)
    l = Activation('relu')(l)
    return BatchNormalization(axis=3, mode=bn_mode)(l)


def res_layer_bottleneck(in_layer, out_dim, bn_dim, activation='relu', bn_mode=0, lightweight=False):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    # 1x1 Bottleneck
    if lightweight:
        x = conv_relu(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation)
    else:
        x = conv_relu_bn(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation,
                         bn_mode=bn_mode)
    # 3x3 conv
    x = conv_relu_bn(x, f_size=3, f_channels=bn_dim, stride=1, border='same', activation=activation, bn_mode=bn_mode)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    if not lightweight:
        x = BatchNormalization(axis=3, mode=bn_mode)(x)
    x = merge([x, in_layer], mode='sum')
    return Activation(activation)(x)


def res_layer_bottleneck_lrelu(in_layer, out_dim, bn_dim, bn_mode=0, lightweight=False):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    # 1x1 Bottleneck
    if lightweight:
        x = conv_lrelu(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same')
    else:
        x = conv_lrelu_bn(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', bn_mode=bn_mode)
    # 3x3 conv
    x = conv_lrelu_bn(x, f_size=3, f_channels=bn_dim, stride=1, border='same', bn_mode=bn_mode)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    if not lightweight:
        x = BatchNormalization(axis=3, mode=bn_mode)(x)
    x = merge([x, in_layer], mode='sum')
    return lrelu(x)


def lrelu(x, alpha=0.2):
    return LeakyReLU(alpha=alpha)(x)


def Generator(input_shape, load_weights=False, big_f=False, w_outter=False, num_res=8):

    # Build the model
    input_gen = Input(shape=input_shape)
    decoded, _ = ToonGenerator(input_gen, big_f=big_f, outter=w_outter, num_res_layers=num_res)
    generator = Model(input_gen, decoded)
    generator.name = make_name('ToonGenerator', w_outter=w_outter, big_f=big_f, num_res=num_res)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mse', optimizer=optimizer)
    return generator


def Encoder(input_shape, load_weights=False, train=False, big_f=False, num_res=8):

    # Build encoder and generator
    input_gen = Input(shape=input_shape)
    decoded, encoded = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res, outter=False)
    encoder = Model(input_gen, encoded)
    generator = Model(input_gen, decoded)
    encoder.name = make_name('ToonEncoder', big_f=big_f, num_res=num_res)
    generator.name = make_name('EncGenTrain', big_f=big_f, num_res=num_res)

    # Load weights
    if load_weights:
        encoder.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if train:
        generator.compile(loss='mse', optimizer=optimizer)
    else:
        encoder.compile(loss='mse', optimizer=optimizer)
    return encoder, generator


def Discriminator(input_shape, load_weights=False, big_f=False, train=True, layer=None, withx=False, num_res=8):

    # Build the model
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res)
    if layer:
        discriminator = Model(input_disc, output=[layer_activations[layer], dis_out])
    else:
        discriminator = Model(input_disc, dis_out)
    make_trainable(discriminator, train)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res)

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def GAN(input_shape, load_weights=False, big_f=False, w_outter=False, recon_weight=5.0, layer=None, withx=False, num_res=8):

    # Build Generator
    input_gen = Input(shape=input_shape)
    gen_out, _ = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res, outter=w_outter)
    generator = Model(input_gen, gen_out)
    generator.name = make_name('ToonGenerator', w_outter=w_outter, big_f=big_f, num_res=num_res)

    # Build Discriminator
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res)
    if layer:
        discriminator = Model(input_disc, output=[layer_activations[layer], dis_out])
    else:
        discriminator = Model(input_disc, output=dis_out)
    make_trainable(discriminator, False)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Build GAN
    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    if withx:
        disc_in = merge([im_input, im_recon], mode='concat')
    else:
        disc_in = im_recon
    disc_out = discriminator(disc_in)
    gan = Model(input=im_input, output=disc_out + [im_recon])

    # Compile the model
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    disc_weight = 1.0
    if layer:
        gan.compile(loss=['mse', 'binary_crossentropy', 'mse'],
                    loss_weights=[1.0, disc_weight, recon_weight],
                    optimizer=optimizer)
    else:
        gan.compile(loss=['binary_crossentropy', 'mse'],
                    loss_weights=[disc_weight, recon_weight],
                    optimizer=optimizer)
    gan.name = make_name('ToonGAN', w_outter=w_outter, layer=layer, with_x=withx)
    return gan, generator, discriminator


def GANwEncoder(input_shape, load_weights=False, big_f=False, w_outter=False, recon_weight=5.0, withx=False, num_res_g=8, num_res_d=8):

    # Build Generator
    input_gen = Input(shape=input_shape)
    gen_out, _ = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res_g, outter=w_outter)
    generator = Model(input_gen, gen_out)
    generator.name = make_name('ToonGenerator', w_outter=w_outter, big_f=big_f, num_res=num_res_g)

    # Build Discriminator
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, _ = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res_d)
    discriminator = Model(input_disc, output=dis_out)
    make_trainable(discriminator, False)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res_d)

    # Build Encoder
    input_encoder = Input(shape=input_shape)
    _, encoder_out = ToonGenerator(input_encoder, big_f=big_f, num_res_layers=num_res_g, outter=False)
    encoder = Model(input_encoder, output=encoder_out)
    make_trainable(encoder, False)
    encoder.name = make_name('ToonEncoder', big_f=big_f, num_res=num_res_g)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))
        encoder.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))

    # Build GAN
    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    if withx:
        disc_in = merge([im_input, im_recon], mode='concat')
    else:
        disc_in = im_recon
    disc_out = discriminator(disc_in)
    encoder_out = encoder(im_recon)
    gan = Model(input=im_input, output=[disc_out, encoder_out, im_recon])

    # Compile model
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    disc_weight = 1.0
    gan.compile(loss=['binary_crossentropy', 'mse', 'mse'],
                loss_weights=[disc_weight, 2.0, recon_weight],
                optimizer=optimizer)
    return gan, generator, discriminator


def make_name(net_name, w_outter=None, layer=None, with_x=None, big_f=None, num_res=None):
    if w_outter:
        net_name = "{}_wout".format(net_name)
    if layer:
        net_name = "{}_L{}".format(net_name, layer)
    if with_x:
        net_name = "{}_wx".format(net_name)
    if big_f:
        net_name = "{}_bigF".format(net_name)
    if num_res:
        net_name = "{}_nr{}".format(net_name, num_res)
    return net_name


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
