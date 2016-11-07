import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, merge, Dense, GlobalAveragePooling2D, \
    Lambda, Flatten, Dropout, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import mae

from constants import MODEL_DIR

F_DIMS = [64, 96, 160, 256, 512]
BF_DIMS = [64, 128, 256, 512, 1024, 2048]
NOISE_CHANNELS = [2, 4, 8, 16, 32, 64, 100]


def ToonGenerator(x, out_activation='tanh', num_res_layers=0, activation='relu', num_layers=5):
    f_dims = BF_DIMS[:num_layers]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)

    for l in range(0, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)

    encoded = x

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer(x, f_dims[num_layers - 1], activation=activation, noise_ch=NOISE_CHANNELS[num_layers])

    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='same', activation=activation)

    for l in range(0, num_layers):
        with tf.name_scope('deconv_{}'.format(l + 1)):
            x = up_conv_act_bn_noise(x, f_size=4, f_channels=f_dims[num_layers - l - 1], activation=activation,
                                     noise_ch=NOISE_CHANNELS[num_layers - l])

    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonDiscriminator(x, num_res_layers=0, activation='lrelu', num_layers=5, noise=None, out_activation='tanh'):
    if noise:
        x = GaussianNoise(sigma=K.get_value(noise))(x)

    f_dims = BF_DIMS[:num_layers]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)

    for l in range(0, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)

    encoded = x

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer(x, f_dims[num_layers - 1], activation=activation)

    for l in range(0, num_layers):
        with tf.name_scope('deconv_{}'.format(l + 1)):
            x = up_conv_act_bn_noise(x, f_size=4, f_channels=f_dims[num_layers - l - 1], activation=activation)

    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonGenerator_old(in_layer, out_activation='tanh', num_res_layers=8, big_f=True, outter=False, activation='relu'):
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
        x = conv_act(in_layer, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
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


def ToonDiscriminator_old(in_layer, num_res_layers=8, big_f=False, p_wise_out=False, activation='lrelu', noise=None):
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

    if noise:
        in_layer = GaussianNoise(sigma=K.get_value(noise))(in_layer)

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


def add_noise_planes(x, n_chan):
    def add_noise(x):
        layer_shape = K.int_shape(x)
        noise = K.random_normal(shape=layer_shape[:3] + (n_chan,), mean=0., std=1.0)
        return merge([x, noise], mode='concat')

    def add_noise_output_shape(input_shape):
        shape = list(input_shape)
        shape[-1] += n_chan
        return tuple(shape)

    x = Lambda(add_noise, output_shape=add_noise_output_shape)(x)
    return x


def up_conv_act(layer_in, f_size, f_channels, activation='relu'):
    def resize(x):
        return K.resize_images(x, height_factor=2, width_factor=2, dim_ordering=K.image_dim_ordering())

    def resize_output_shape(input_shape):
        shape = list(input_shape)
        shape[1] *= 2
        shape[2] *= 2
        return tuple(shape)

    x = Lambda(resize, output_shape=resize_output_shape)(layer_in)
    # x = UpSampling2D()(layer_in)
    x = conv_act(x, f_size=f_size, f_channels=f_channels, stride=1, border='same', activation=activation)
    return x


def up_conv_act_bn_noise(layer_in, f_size, f_channels, activation='relu', noise_ch=None):
    def resize(x):
        return K.resize_images(x, height_factor=2, width_factor=2, dim_ordering=K.image_dim_ordering())

    def resize_output_shape(input_shape):
        shape = list(input_shape)
        shape[1] *= 2
        shape[2] *= 2
        return tuple(shape)

    x = Lambda(resize, output_shape=resize_output_shape)(layer_in)
    if noise_ch:
        x = add_noise_planes(x, noise_ch)
    x = conv_act_bn(x, f_size=f_size, f_channels=f_channels, stride=1, border='same', activation=activation)
    return x


def conv_act_bn(layer_in, f_size, f_channels, stride, border='valid', activation='relu', regularizer=None):
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
                      init='he_normal',
                      W_regularizer=regularizer)(layer_in)
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


def res_layer(in_layer, out_dim, activation='relu', noise_ch=None):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    x = in_layer
    if noise_ch:
        x = add_noise_planes(x, noise_ch)
    x = conv_act_bn(x, f_size=3, f_channels=out_dim, stride=1, border='same', activation=activation)
    x = merge([x, in_layer], mode='sum')
    return my_activation(x, type=activation)


def my_activation(x, type='relu', alpha=0.2):
    if type == 'relu':
        return Activation(type)(x)
    elif type == 'lrelu':
        return LeakyReLU(alpha=alpha)(x)
    else:
        raise ValueError('Activation type {} not supported'.format(type))


def sub(a, b):
    return merge([a, b], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])


def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred), axis=-1)


def l2_mb(y_true, y_pred):  # Idea: could pass in margin during training (similar to noise thingie)
    # return -K.mean(K.maximum(10.0-K.abs(y_pred), 0), axis=-1)
    return K.mean(K.minimum(K.square(y_pred), 2.0), axis=-1)


def l2_ms(y_true, y_pred):
    # return -K.mean(K.maximum(0.2-K.square(y_pred), 0), axis=-1)
    return K.mean(K.minimum(K.square(y_pred), 0.2), axis=-1)


def Classifier(input_shape, batch_size=128, num_layers=4, num_res=0, num_classes=1000, net_load_name=None,
               compile_model=True, use_gen=False):
    # Build encoder
    input_im = Input(batch_shape=(batch_size,) + input_shape)
    if use_gen:
        decoded, encoded = ToonGenerator(input_im, num_layers=num_layers, num_res_layers=num_res)
    else:
        decoded, encoded = ToonDiscriminator(input_im, num_layers=num_layers, num_res_layers=num_res)
    encoder = Model(input_im, encoded)
    discriminator = Model(input_im, decoded)
    if net_load_name:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_load_name)))
        make_trainable(encoder, False)

    # Build classifier
    im_input = Input(batch_shape=(batch_size,) + input_shape)
    enc_out = encoder(im_input)

    # Dense Layers
    x = Flatten()(enc_out)
    x = Dense(2048, init='he_normal')(x)
    x = Dropout(0.25)(x)
    x = my_activation(x, type='relu')
    x = BatchNormalization(axis=1)(x)
    prediction = Dense(num_classes, init='he_normal', activation='softmax')(x)

    classifier = Model(input=im_input, output=prediction)
    classifier.name = 'Classifier_{}'.format(net_load_name)

    if compile_model:
        optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        classifier.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return classifier


def Generator(input_shape, load_weights=False, num_layers=4, batch_size=128, num_res=0):
    # Build the model
    input_gen = Input(batch_shape=(batch_size,) + input_shape)
    decoded, _ = ToonGenerator(input_gen, num_layers=num_layers, num_res_layers=num_res)
    generator = Model(input_gen, decoded)
    generator.name = make_name('ToonGenerator', num_layers=num_layers, num_res=num_res)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mse', optimizer=optimizer)
    return generator


def Discriminator(input_shape, num_layers=4, load_weights=False, train=True, noise=None, num_res=0):
    # Build the model
    input_disc = Input(shape=input_shape)
    dis_out, _ = ToonDiscriminator(input_disc, num_layers=num_layers, noise=noise, num_res_layers=num_res)
    discriminator = Model(input_disc, dis_out)
    make_trainable(discriminator, train)
    discriminator.name = make_name('ToonDiscriminator', num_layers=num_layers, num_res=num_res)

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='mse', optimizer=optimizer)
    return discriminator


def EBGAN(input_shape, batch_size=128, load_weights=False, num_layers_g=4, num_layers_d=4, noise=None, train_disc=True,
          r_weight=20.0, d_weight=1.0, num_res=0):
    # Build Generator
    input_gen = Input(batch_shape=(batch_size,) + input_shape)
    gen_out, _ = ToonGenerator(input_gen, num_layers=num_layers_g, num_res_layers=num_res)
    generator = Model(input_gen, gen_out)
    generator.name = make_name('ToonGenerator', num_layers=num_layers_g, num_res=num_res)
    if train_disc:
        make_trainable(generator, False)

    # Build Discriminator
    input_disc = Input(shape=input_shape)
    dis_out, _ = ToonDiscriminator(input_disc, num_layers=num_layers_d, noise=noise, num_res_layers=num_res)
    discriminator = Model(input_disc, output=dis_out)
    discriminator.name = make_name('ToonDiscriminator', num_layers=num_layers_d, num_res=num_res)
    if not train_disc:
        make_trainable(discriminator, False)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    # Build GAN
    x_input = Input(batch_shape=(batch_size,) + input_shape)
    y_input = Input(shape=input_shape)
    g_x = generator(x_input)
    d_g_x = discriminator(g_x)
    d_y = discriminator(y_input)
    l2 = sub(d_g_x, g_x)
    l3 = sub(d_g_x, d_y)
    if train_disc:
        l1 = sub(d_y, y_input)
        gan = Model(input=[x_input, y_input], output=[l1, l2, l3])
        gan.compile(loss=['mse'] * 3, loss_weights=[r_weight, -1.0, -d_weight], optimizer=optimizer)
        gan.name = make_name('dGAN', num_layers=[num_layers_d, num_layers_g], num_res=num_res, r_weight=r_weight,
                             d_weight=d_weight)
    else:
        gan = Model(input=[x_input, y_input], output=[l2, l3])
        gan.compile(loss=['mse'] * 2, loss_weights=[1.0, d_weight], optimizer=optimizer)
        gan.name = make_name('gGAN', num_layers=[num_layers_d, num_layers_g], num_res=num_res, r_weight=r_weight,
                             d_weight=d_weight)

    return gan, generator, discriminator


def EBGAN2(input_shape, batch_size=128, load_weights=False, num_layers_g=4, num_layers_d=4, noise=None, train_disc=True,
          r_weight=20.0, d_weight=1.0, num_res=0):
    # Build Generator
    input_gen = Input(batch_shape=(batch_size,) + input_shape)
    gen_out, gen_enc = ToonGenerator(input_gen, num_layers=num_layers_g, num_res_layers=num_res)
    generator = Model(input_gen, [gen_out, gen_enc])
    generator.name = make_name('ToonGenerator', num_layers=num_layers_g, num_res=num_res)
    if train_disc:
        make_trainable(generator, False)

    # Build Discriminator
    input_disc = Input(shape=input_shape)
    dis_out, disc_enc = ToonDiscriminator(input_disc, num_layers=num_layers_d, noise=noise, num_res_layers=num_res, activation='relu')
    discriminator = Model(input_disc, output=[dis_out, disc_enc])
    discriminator.name = make_name('ToonDiscriminator', num_layers=num_layers_d, num_res=num_res)
    if not train_disc:
        make_trainable(discriminator, False)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    # Build GAN
    x_input = Input(batch_shape=(batch_size,) + input_shape)
    y_input = Input(batch_shape=(batch_size,) + input_shape)
    g_x, ge_x = generator(x_input)
    _, ge_y = generator(y_input)
    d_g_x, de_g_x = discriminator(g_x)
    d_y, de_y = discriminator(y_input)
    l1 = sub(de_g_x, de_y)    # before
    l5 = sub(d_g_x, d_y)  # 19931
    l2 = sub(d_g_x, g_x)
    if train_disc:
        l3 = sub(d_y, y_input)
        gan = Model(input=[x_input, y_input], output=[l1, l2, l3, l5])
        gan.compile(loss=[l2_loss, l2_ms, l2_loss, l2_loss], loss_weights=[-1.0, -d_weight, d_weight, r_weight], optimizer=optimizer)
        gan.name = make_name('dGAN2', num_layers=[num_layers_d, num_layers_g], num_res=num_res, r_weight=r_weight,
                             d_weight=d_weight)
    else:
        l4 = sub(g_x, y_input)
        # l4 = sub(ge_x, ge_y)
        gan = Model(input=[x_input, y_input], output=[l1, l2, l4, l5])
        gan.compile(loss=[l2_loss, l2_ms, l2_loss, l2_loss], loss_weights=[1.0, d_weight, r_weight, r_weight], optimizer=optimizer)
        gan.name = make_name('gGAN2', num_layers=[num_layers_d, num_layers_g], num_res=num_res, r_weight=r_weight,
                             d_weight=d_weight)

    return gan, generator, discriminator


def Generator_old(input_shape, load_weights=False, big_f=False, w_outter=False, num_res=8, activation='relu'):
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


def Encoder(input_shape, load_weights=False, train=False, big_f=False, num_res=8, layers=[5], activation='relu'):
    # Build encoder and generator
    input_gen = Input(shape=input_shape)
    decoded, enc_layers = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res, outter=False,
                                        activation=activation)
    enc_out = [enc_layers[l - 1] for l in layers]
    encoder = Model(input_gen, enc_out)
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
        encoder.compile(loss=['mse'] * len(layers), optimizer=optimizer)
    return encoder, generator


def Discriminator_old(input_shape, load_weights=False, big_f=False, train=True, layers=None, withx=False, num_res=8,
                      p_wise_out=False, noise=None):
    # Build the model
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res,
                                                   p_wise_out=p_wise_out, noise=noise)
    if layers:
        disc_layers_out = [layer_activations[l - 1] for l in layers]
        discriminator = Model(input_disc, output=disc_layers_out + [dis_out])
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
    if train:
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    else:
        discriminator.compile(loss=len(layers) * ['mse'] + ['binary_crossentropy'], optimizer=optimizer)
    return discriminator


def GANwGen(input_shape, load_weights=False, big_f=False, recon_weight=5.0, withx=False, num_res_g=8, num_res_d=8,
            enc_weight=1.0, layers=[5], learning_rate=0.0002, w_outter=False, p_wise_out=False, activation='relu'):
    # Build Generator
    input_gen = Input(shape=input_shape)
    gen_out, gen_layers = ToonGenerator(input_gen, big_f=big_f, num_res_layers=num_res_g, outter=w_outter,
                                        activation=activation)
    generator = Model(input_gen, gen_out)
    gen_enc_out = [gen_layers[l - 1] for l in layers]
    gen_enc = Model(input_gen, gen_enc_out)
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
    enc_out = [enc_layers[l - 1] for l in layers]
    enc_on_gan = Model(input_encoder, output=enc_out)
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
    gan = Model(input=im_input, output=[disc_out] + encoded + [im_recon])

    # Compile model
    optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss_weights = [1.0] + range(1, len(layers) + 1) * enc_weight + [recon_weight]
    gan.compile(loss=['binary_crossentropy'] + len(layers) * ['mse'] + ['mse'],
                loss_weights=loss_weights,
                optimizer=optimizer)
    return gan, generator, discriminator, gen_enc, enc_on_gan


def GANwDisc(input_shape, load_weights=False, big_f=False, recon_weight=5.0, withx=False, num_res_g=8, num_res_d=8,
             enc_weight=1.0, layers=[5], learning_rate=0.0002, w_outter=False, p_wise_out=False, activation='relu',
             noise=None):
    # Build Generator
    input_gen = Input(shape=input_shape)
    gen_out, _ = ToonGenerator(input_gen, num_res_layers=num_res_g, outter=w_outter,
                               activation=activation)
    generator = Model(input_gen, gen_out)
    generator.name = make_name('ToonGenerator', num_res=num_res_g, w_outter=w_outter,
                               activation=activation)

    # Build Discriminator
    if withx:
        input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    else:
        input_disc = Input(shape=input_shape)
    dis_out, disc_layers = ToonDiscriminator(input_disc, big_f=big_f, num_res_layers=num_res_d, p_wise_out=p_wise_out,
                                             noise=noise)
    disc_layers_out = [disc_layers[l - 1] for l in layers]
    discriminator = Model(input_disc, output=disc_layers_out + [dis_out])
    make_trainable(discriminator, False)
    discriminator.name = make_name('ToonDiscriminator', with_x=withx, big_f=big_f, num_res=num_res_d,
                                   p_wise_out=p_wise_out)

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

    # Compile model
    optimizer = Adam(lr=learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss_weights = [enc_weight * l for l in range(1, len(layers) + 1)] + [1.0, recon_weight]
    gan.compile(loss=len(layers) * ['mse'] + ['binary_crossentropy'] + ['mse'],
                loss_weights=loss_weights,
                optimizer=optimizer)
    return gan, generator, discriminator


def make_name(net_name, w_outter=None, layer=None, with_x=None, big_f=None, num_res=None, p_wise_out=None,
              activation=None, num_layers=None, r_weight=None, d_weight=None, noise=None):
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
    if num_layers:
        net_name = "{}_nl{}".format(net_name, num_layers)
    if r_weight:
        net_name = "{}_rw{}".format(net_name, r_weight)
    if d_weight:
        net_name = "{}_dw{}".format(net_name, d_weight)
    if noise:
        net_name = "{}_noise".format(net_name)
    return net_name


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def disc_data(X, Y, Yd, p_wise=False, with_x=False):
    if with_x:
        Xd = np.concatenate((np.concatenate((X, Y), axis=3), np.concatenate((X, Yd), axis=3)))
    else:
        Xd = np.concatenate((Y, Yd))

    if p_wise:
        yd = np.concatenate((np.ones((len(Y), 4, 4, 1)), np.zeros((len(Y), 4, 4, 1))), axis=0)
    else:
        yd = np.zeros((len(Y) + len(Yd), 1))
        yd[:len(Y)] = 1
    return Xd, yd
