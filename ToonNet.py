import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, merge, Dense, Lambda, Flatten, Dropout, \
    GaussianNoise, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from constants import MODEL_DIR

F_DIMS = [64, 128, 256, 512, 1024, 2048]
F_G_DIMS = [64, 96, 160, 256, 416, 672, 1088]

NOISE_CHANNELS = [2, 4, 8, 16, 32, 64, 100]


def ToonGenTransp(x, out_activation='tanh', activation='relu', num_layers=5, batch_size=128):
    f_dims = F_DIMS[:num_layers + 1]
    x = add_noise_planes(x, 1)
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
    l_dims = [K.int_shape(x)[1]]

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation=activation)
            # x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)
            l_dims += [K.int_shape(x)[1]]

    x = conv_act_bn(x, f_size=4, f_channels=f_dims[num_layers], stride=1, border='valid', activation=activation)
    encoded = x
    l_dims += [K.int_shape(x)[1]]
    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    # x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='same', activation=activation)
    x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - 1], out_dim=l_dims[num_layers - 1],
                       batch_size=batch_size, activation=activation, border='same')

    for l in range(1, num_layers):
        with tf.name_scope('conv_transp_{}'.format(l + 1)):
            # x = up_conv_act_bn_noise(x, f_size=4, f_channels=f_dims[num_layers - l - 1], activation=activation,
            #                          noise_ch=NOISE_CHANNELS[num_layers - l])
            x = add_noise_planes(x, NOISE_CHANNELS[num_layers - l])
            x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - l - 1], out_dim=l_dims[num_layers - l - 1],
                               batch_size=batch_size, activation=activation)

    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonGen(x, out_activation='tanh', activation='relu', num_layers=5, batch_size=128):
    f_dims = F_DIMS[:num_layers + 1]
    x = add_noise_planes(x, 1)
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
    l_dims = [K.int_shape(x)[1]]

    for l in range(0, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)
            # x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)
            l_dims += [K.int_shape(x)[1]]

    encoded = x
    l_dims += [K.int_shape(x)[1]]
    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='same', activation=activation)

    for l in range(0, num_layers):
        with tf.name_scope('conv_transp_{}'.format(l + 1)):
            x = up_conv_act_bn_noise(x, f_size=4, f_channels=f_dims[num_layers - l - 1], activation=activation,
                                     noise_ch=NOISE_CHANNELS[num_layers - l])

    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonDisc(x, activation='lrelu', num_layers=5, noise=None):
    if noise:
        x = GaussianNoise(sigma=K.get_value(noise))(x)

    f_dims = F_DIMS[:num_layers]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='valid', activation=activation)

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation=activation)

    x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='valid', activation=activation)
    p_out = x

    # p_out = Convolution2D(1, 1, 1, subsample=(1, 1), init='he_normal', activation='sigmoid')(x)
    x = Flatten()(x)
    x = Dense(2048, init='he_normal')(x)
    x = my_activation(x, type=activation)
    x = BatchNormalization(axis=1)(x)
    x = Dense(2048, init='he_normal')(x)
    x = my_activation(x, type=activation)
    x = BatchNormalization(axis=1)(x)
    d_out = Dense(1, init='he_normal', activation='sigmoid')(x)

    return d_out, p_out


def GAN(input_shape, batch_size=128, num_layers=4, load_weights=False, noise=None):
    gen_in_shape = (batch_size,) + input_shape[:2] + (4,)

    # Build Generator
    input_gen = Input(batch_shape=gen_in_shape)
    g_out, _ = ToonGen(input_gen, num_layers=num_layers, batch_size=batch_size)
    generator = Model(input_gen, g_out)
    generator.name = make_name('ToonGen', num_layers=num_layers)

    # Build Discriminator
    input_disc = Input(shape=input_shape)
    d_out, d_enc = ToonDisc(input_disc, num_layers=num_layers, activation='relu', noise=noise)
    discriminator = Model(input_disc, output=[d_out, d_enc])
    discriminator.name = make_name('ToonDisc', num_layers=num_layers)
    make_trainable(discriminator, False)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Build GAN
    g_input = Input(batch_shape=gen_in_shape)
    img_input = Input(shape=input_shape)
    g_x = generator(g_input)
    d_g_x, de_g_x = discriminator(g_x)
    _, de_y = discriminator(img_input)

    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    gan = Model(input=[g_input, img_input], output=[d_g_x, g_x, de_g_x])
    gan.compile(loss=['binary_crossentropy', 'mse', 'mae'], loss_weights=[1.0, 20.0, 1.0],
                optimizer=optimizer)
    gan.name = make_name('ToonGAN', num_layers=num_layers)

    return gan, generator, discriminator


def Gen(input_shape, load_weights=False, num_layers=4, batch_size=128):
    # Build the model
    input_gen = Input(batch_shape=(batch_size,) + input_shape[:2] + (4,))

    decoded, _ = ToonGen(input_gen, num_layers=num_layers, batch_size=batch_size)
    generator = Model(input_gen, decoded)
    generator.name = make_name('ToonGenerator', num_layers=num_layers)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mse', optimizer=optimizer)
    return generator


def Disc(input_shape, load_weights=False, num_layers=4, noise=None):
    # Build the model
    input_gen = Input(shape=input_shape)

    d_out, dp_out = ToonDisc(input_gen, num_layers=num_layers, activation='relu', noise=noise)
    discriminator = Model(input_gen, d_out)
    discriminator.name = make_name('ToonDiscriminator', num_layers=num_layers)

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


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


def conv_transp_bn(layer_in, f_size, f_channels, out_dim, batch_size, stride=2, border='valid', activation='relu'):
    """Wrapper for upconvolution layer including batchnormalization.
    Args:
        layer_in: Input to this layers
        f_size: Size of the filers
        f_channels: Number of output channels
        out_dim: Dimension/shape of the output
        batch_size:
        stride: Used stride for the convolution
        border: 'valid' or 'same'
    Returns:
        Result of upconvolution and batchnormalization
    """
    x = Deconvolution2D(f_channels, f_size, f_size,
                        output_shape=(batch_size, out_dim, out_dim, f_channels),
                        border_mode=border,
                        subsample=(stride, stride),
                        init='he_normal')(layer_in)
    x = my_activation(x, type=activation)
    return BatchNormalization(axis=3)(x)


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


def GAN2(input_shape, batch_size=128, num_layers=5, load_weights=False, noise=None, train_disc=True):
    gen_in_shape = (batch_size,) + input_shape[:2] + (4,)

    # Build Generator
    input_gen = Input(batch_shape=gen_in_shape)
    g_out, _ = ToonGen(input_gen, num_layers=num_layers, batch_size=batch_size)
    generator = Model(input_gen, g_out)
    generator.name = make_name('ToonGen', num_layers=num_layers)
    if train_disc:
        make_trainable(generator, False)

    # Build Discriminator
    input_disc = Input(shape=input_shape)
    d_out, d_enc = ToonDisc(input_disc, num_layers=num_layers, activation='relu', noise=noise)
    discriminator = Model(input_disc, output=[d_out, d_enc])
    discriminator.name = make_name('ToonDisc', num_layers=num_layers)
    if not train_disc:
        make_trainable(discriminator, False)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # Build GAN
    g_input = Input(batch_shape=gen_in_shape)
    img_input = Input(shape=input_shape)
    g_x = generator(g_input)
    d_g_x, de_g_x = discriminator(g_x)
    d_y, de_y = discriminator(img_input)

    if train_disc:
        disc_pred = merge([d_g_x, d_y], mode='concat', concat_axis=0)
        gan = Model(input=[g_input, img_input], output=[disc_pred])
        gan.compile(loss=ld_merged, loss_weights=1.0, optimizer=optimizer)
        gan.name = make_name('dGAN', num_layers=num_layers)
    else:
        l1 = sub(g_x, img_input)
        gan = Model(input=[g_input, img_input], output=[d_g_x, l1])
        gan.compile(loss=[ld1, l2], loss_weights=[1.0, 1.0], optimizer=optimizer)
        gan.name = make_name('gGAN', num_layers=num_layers)

    return gan, generator, discriminator


def ld_merged(y_true, y_pred):
    t_shape = K.int_shape(y_pred)
    t_shape[0] /= 2
    target = K.concatenate([K.zeros(shape=t_shape), K.ones(shape=t_shape)], axis=0)
    return K.mean(K.binary_crossentropy(y_pred, target), axis=-1)


def ld0(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, K.zeros_like(y_pred)), axis=-1)


def ld1(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, K.ones_like(y_pred)), axis=-1)


def l2_margin(y_true, y_pred):
    return -K.maximum(1.0 - K.mean(K.square(y_pred), axis=-1), 0)


def l2(y_true, y_pred):
    return K.mean(K.square(y_pred), axis=-1)


def l1_margin(y_true, y_pred):
    return -K.maximum(1.0 - K.mean(K.abs(y_pred-y_true), axis=-1), 0)


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


def disc_data(X, Y, Yd, with_x=False):
    if with_x:
        Xd = np.concatenate((np.concatenate((X, Y), axis=3), np.concatenate((X, Yd), axis=3)))
    else:
        Xd = np.concatenate((Y, Yd))

    yd = np.zeros((len(Y) + len(Yd), 1))
    yd[:len(Y)] = 1
    return Xd, yd


def gen_data(im, edge):
    X = np.concatenate((im, edge), axis=3)
    return X
