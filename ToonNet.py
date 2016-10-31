import os

import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, merge, Dense, UpSampling2D, \
    GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from constants import MODEL_DIR

F_DIMS = [64, 96, 160, 256, 512]
BF_DIMS = [64, 128, 256, 512, 1024]


def ToonGenerator(in_layer, out_activation='tanh', num_res_layers=8, big_f=False, outter=False, activation='relu'):
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
        x = conv_act(in_layer, f_size=3, f_channels=32, stride=1, border='same', activation=activation)
        l1 = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same', activation=activation)
        if outter:
            ol1 = outter_connection(l1, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_act(l1, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        l2 = conv_act_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', activation=activation)
        if outter:
            ol2 = outter_connection(l2, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_act(l2, f_size=3, f_channels=f_dims[1], stride=1, border='same', activation=activation)
        l3 = conv_act_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same', activation=activation)
        if outter:
            ol3 = outter_connection(l3, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_act(l3, f_size=3, f_channels=f_dims[2], stride=1, border='same', activation=activation)
        l4 = conv_act_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same', activation=activation)
        if outter:
            ol4 = outter_connection(l4, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_act(l4, f_size=3, f_channels=f_dims[3], stride=1, border='same', activation=activation)
        x = conv_act_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same', activation=activation)
        if outter:
            ol5 = outter_connection(x, f_dims[4])

    l5 = x
    layers = [l1, l2, l3, l4, l5]

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[1], activation=activation, lightweight=True)
    if outter:
        x = merge([x, ol5], mode='sum')

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = up_conv_act(x, f_size=3, f_channels=f_dims[3], activation=activation)
        if outter:
            x = merge([x, ol4], mode='sum')
        x = conv_act_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', activation=activation)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = up_conv_act(x, f_size=3, f_channels=f_dims[2], activation=activation)
        if outter:
            x = merge([x, ol3], mode='sum')
        x = conv_act_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', activation=activation)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = up_conv_act(x, f_size=3, f_channels=f_dims[1], activation=activation)
        if outter:
            x = merge([x, ol2], mode='sum')
        x = conv_act_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', activation=activation)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = up_conv_act(x, f_size=3, f_channels=f_dims[0], activation=activation)
        if outter:
            x = merge([x, ol1], mode='sum')
        x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = up_conv_act(x, f_size=3, f_channels=32, activation=activation)
        x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
        decoded = Activation(out_activation)(x)

    return decoded, layers


def ToonDiscriminator(in_layer, num_res_layers=8, big_f=False, p_wise_out=False, activation='lrelu'):
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
        x = conv_act(in_layer, f_size=3, f_channels=32, stride=1, border='same', activation=activation)
        l1 = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same', activation=activation)

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_act(l1, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
        l2 = conv_act_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', activation=activation)

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_act(l2, f_size=3, f_channels=f_dims[1], stride=1, border='same', activation=activation)
        l3 = conv_act_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same', activation=activation)

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_act(l3, f_size=3, f_channels=f_dims[2], stride=1, border='same', activation=activation)
        l4 = conv_act_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same', activation=activation)

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_act(l4, f_size=3, f_channels=f_dims[3], stride=1, border='same', activation=activation)
        l5 = conv_act_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same', activation=activation)

    layer_activations = [l1, l2, l3, l4, l5]
    x = l5

    # Res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[1], lightweight=True, activation=activation)

    if p_wise_out:
        x = Convolution2D(1, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal', activation='sigmoid')(x)
    else:
        # Fully connected layer
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, init='he_normal')(x)
        x = my_activation(x, type='lrelu')
        x = BatchNormalization(axis=1)(x)
        layer_activations.append(x)
        x = Dense(1, init='he_normal')(x)
        x = Activation('sigmoid')(x)

    return x, layer_activations


def up_conv_act(layer_in, f_size, f_channels, activation='relu'):
    x = K.resize_images(layer_in, height_factor=2.0, width_factor=2.0, dim_ordering='tf')
    # x = UpSampling2D()(layer_in)
    x = conv_act(x, f_size=f_size, f_channels=f_channels, stride=1, border='same', activation=activation)
    return x


def conv_act_bn(layer_in, f_size, f_channels, stride, border='valid', activation='relu'):
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
    x = my_activation(x, type=activation)
    return BatchNormalization(axis=3)(x)


def conv_act(layer_in, f_size, f_channels, stride, border='valid', activation='relu'):
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
    return my_activation(x, type=activation)


def outter_connection(layer_in, f_channels):
    """Wrapper for 1x1 convolutions used on the outer layers.

    Args:
        layer_in: Input to this layer
        f_channels: Number of channels for the output (usually the same as input)

    Returns:
        Result of convolution followed by batchnorm and leakyRelu
    """
    l = Convolution2D(f_channels, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal')(layer_in)
    l = Activation('relu')(l)
    return BatchNormalization(axis=3)(l)


def res_layer_bottleneck(in_layer, out_dim, bn_dim, activation='relu', lightweight=False):
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
        x = conv_act(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation)
    else:
        x = conv_act_bn(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation)
    # 3x3 conv
    x = conv_act_bn(x, f_size=3, f_channels=bn_dim, stride=1, border='same', activation=activation)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    if not lightweight:
        x = BatchNormalization(axis=3)(x)
    x = merge([x, in_layer], mode='sum')
    return my_activation(x, type=activation)


def my_activation(x, type='relu', alpha=0.3):
    if type == 'relu':
        return Activation(type)(x)
    elif type == 'lrelu':
        return LeakyReLU(alpha=alpha)(x)
    else:
        raise ValueError('Activation type {} not supported'.format(type))


def Generator(input_shape, load_weights=False, big_f=False, w_outter=False, num_res=8, activation='relu'):
    # Build the model
    input_gen = Input(shape=input_shape)
    decoded, _ = ToonGenerator(input_gen, big_f=big_f, outter=w_outter, num_res_layers=num_res, activation=activation)
    generator = Model(input_gen, decoded)
    generator.name = make_name('ToonGenerator', w_outter=w_outter, big_f=big_f, num_res=num_res, activation=activation)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mse', optimizer=optimizer)
    return generator


def Encoder(input_shape, load_weights=False, train=False, big_f=False, num_res=8, layer=5, activation='relu'):
    # Build encoder and generator
    input_gen = Input(shape=input_shape)
    decoded, enc_layers = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res, outter=False,
                                        activation=activation)
    encoder = Model(input_gen, enc_layers[layer - 1])
    generator = Model(input_gen, decoded)
    encoder.name = make_name('ToonEncoder', big_f=big_f, num_res=num_res, activation=activation)
    generator.name = make_name('EncGenTrain', big_f=big_f, num_res=num_res, activation=activation)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    if train:
        generator.compile(loss='mse', optimizer=optimizer)
    else:
        encoder.compile(loss='mse', optimizer=optimizer)
    return encoder, generator


def Discriminator(input_shape, load_weights=False, big_f=False, train=True, layer=None, withx=False, num_res=8,
                  p_wise_out=False):
    # Build the model
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res,
                                                   p_wise_out=p_wise_out)
    if layer:
        discriminator = Model(input_disc, output=[layer_activations[layer], dis_out])
    else:
        discriminator = Model(input_disc, dis_out)
    make_trainable(discriminator, train)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res,
                                   p_wise_out=p_wise_out)

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def GANwGen(input_shape, load_weights=False, big_f=False, recon_weight=5.0, withx=False, num_res_g=8, num_res_d=8,
            enc_weight=1.0, layer=5, learning_rate=0.0002, w_outter=False, p_wise_out=False, activation='relu'):
    # Build Generator
    input_gen = Input(shape=input_shape)
    gen_out, gen_layers = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res_g, outter=w_outter,
                                        activation=activation)
    generator = Model(input_gen, gen_out)
    gen_enc = Model(input_gen, gen_layers[layer - 1])
    gen_enc.name = make_name('ToonGenEnc', big_f=big_f, num_res=num_res_g, w_outter=w_outter, activation=activation)
    generator.name = make_name('ToonGenerator', big_f=big_f, num_res=num_res_g, w_outter=w_outter,
                               activation=activation)

    # Build Discriminator
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, _ = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res_d, p_wise_out=p_wise_out)
    discriminator = Model(input_disc, output=dis_out)
    make_trainable(discriminator, False)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res_d,
                                   p_wise_out=p_wise_out)

    # Build Encoder
    input_encoder = Input(shape=input_shape)
    _, enc_layers = ToonGenerator(input_encoder, big_f=big_f, num_res_layers=num_res_g, outter=w_outter,
                                  activation=activation)
    enc_on_gan = Model(input_encoder, output=enc_layers[layer - 1])
    make_trainable(enc_on_gan, False)
    enc_on_gan.name = make_name('ToonEncOnGan', big_f=big_f, num_res=num_res_g, w_outter=w_outter,
                                activation=activation)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))
        enc_on_gan.set_weights(gen_enc.get_weights())

    # Build GAN
    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    if withx:
        disc_in = merge([im_input, im_recon], mode='concat')
    else:
        disc_in = im_recon
    disc_out = discriminator(disc_in)
    encoded = enc_on_gan(im_recon)
    gan = Model(input=im_input, output=[disc_out, encoded, im_recon])

    # Compile model
    optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    disc_weight = 1.0
    gan.compile(loss=['binary_crossentropy', 'mse', 'mse'],
                loss_weights=[disc_weight, enc_weight, recon_weight],
                optimizer=optimizer)
    return gan, generator, discriminator, gen_enc, enc_on_gan


def make_name(net_name, w_outter=None, layer=None, with_x=None, big_f=None, num_res=None, p_wise_out=None,
              activation=None):
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
    if p_wise_out:
        net_name = "{}_pwo".format(net_name)
    if activation:
        net_name = "{}_{}".format(net_name, activation)
    return net_name


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
