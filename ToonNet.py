import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Activation, merge, Dense, GlobalAveragePooling2D, \
    Lambda, Flatten, Dropout, GaussianNoise, Deconvolution2D, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from constants import MODEL_DIR

F_DIMS = [64, 128, 256, 512, 1024, 2048]

NOISE_CHANNELS = [2, 4, 8, 16, 32, 64, 100]


def ToonGenTransp(x, out_activation='tanh', activation='relu', num_layers=5, batch_size=128):
    f_dims = F_DIMS[:num_layers+1]
    x = add_noise_planes(x, 1)
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
    l_dims = [K.int_shape(x)[1]]

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation=activation)
            l_dims += [K.int_shape(x)[1]]

    x = conv_act_bn(x, f_size=4, f_channels=f_dims[num_layers], stride=1, border='valid', activation=activation)
    encoded = x
    l_dims += [K.int_shape(x)[1]]
    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - 1], out_dim=l_dims[num_layers - 1],
                       batch_size=batch_size, activation=activation, border='same')

    for l in range(1, num_layers):
        with tf.name_scope('conv_transp_{}'.format(l + 1)):
            x = add_noise_planes(x, NOISE_CHANNELS[num_layers - l])
            x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - l - 1], out_dim=l_dims[num_layers - l - 1],
                               batch_size=batch_size, activation=activation)

    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonGen(x, out_activation='tanh', activation='relu', num_layers=5, batch_size=128):
    num_layers += 1
    f_dims = F_DIMS[:num_layers+1]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
    l_dims = [K.int_shape(x)[1]]

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='same', activation=activation)
            l_dims += [K.int_shape(x)[1]]

    x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='same', activation=activation)
    encoded = x
    l_dims += [K.int_shape(x)[1]]
    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='same', activation=activation)

    for l in range(1, num_layers):
        with tf.name_scope('conv_transp_{}'.format(l + 1)):
            x = up_conv_act_bn_noise(x, f_size=4, f_channels=f_dims[num_layers - l - 1], activation=activation,
                                     noise_ch=NOISE_CHANNELS[num_layers - l])

    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    decoded = Activation(out_activation)(x)

    return decoded, encoded


def ToonGenAE(x, activation='relu', num_layers=5):
    f_dims = F_DIMS[:num_layers+1]
    x = add_noise_planes(x, NOISE_CHANNELS[0])
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation=activation)
    l_dims = [K.int_shape(x)[1]]

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = add_noise_planes(x, NOISE_CHANNELS[l])
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation=activation)
            l_dims += [K.int_shape(x)[1]]

    x = add_noise_planes(x, NOISE_CHANNELS[num_layers])
    x = conv_act_bn(x, f_size=4, f_channels=f_dims[num_layers], stride=1, border='valid', activation=activation)
    encoded = x
    l_dims += [K.int_shape(x)[1]]

    return encoded, l_dims


def ToonEncAE(x, num_layers=5):
    f_dims = F_DIMS[:num_layers+1]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', activation='relu')
    x = SpatialDropout2D(0.5)(x)
    l_dims = [K.int_shape(x)[1]]

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = SpatialDropout2D(0.5)(x)
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation='relu')
            l_dims += [K.int_shape(x)[1]]

    x = conv_act_bn(x, f_size=4, f_channels=f_dims[num_layers], stride=1, border='valid', activation='relu')
    encoded = x
    l_dims += [K.int_shape(x)[1]]

    return encoded, l_dims


def ToonDecoder(x, num_layers, l_dims, batch_size=128):
    f_dims = F_DIMS[:num_layers+1]
    x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - 1], out_dim=l_dims[num_layers - 1],
                       batch_size=batch_size, activation='relu', border='same')

    for l in range(1, num_layers):
        with tf.name_scope('conv_transp_{}'.format(l + 1)):
            x = conv_transp_bn(x, f_size=4, f_channels=f_dims[num_layers - l - 1], out_dim=l_dims[num_layers - l - 1],
                               batch_size=batch_size, activation='relu')

    x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    return Activation('tanh')(x)


def ToonDisc(x, activation='lrelu', num_layers=5):

    x = SpatialDropout2D(p=0.5)(x)

    f_dims = F_DIMS[:num_layers]
    x = conv_act_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='valid', activation=activation)

    for l in range(1, num_layers):
        with tf.name_scope('conv_{}'.format(l + 1)):
            x = SpatialDropout2D(p=0.5)(x)
            x = conv_act_bn(x, f_size=4, f_channels=f_dims[l], stride=2, border='valid', activation=activation)

    encoded = conv_act_bn(x, f_size=3, f_channels=f_dims[num_layers - 1], stride=1, border='valid', activation=activation)
    p_out = conv_act(encoded, f_size=1, f_channels=1, stride=1, border='valid', activation='sigmoid')

    x = Flatten()(encoded)
    x = Dense(2048, init='he_normal')(x)
    x = Dropout(0.5)(x)
    x = my_activation(x, type=activation)
    x = BatchNormalization(axis=1)(x)
    x = Dense(2048, init='he_normal')(x)
    x = Dropout(0.5)(x)
    x = my_activation(x, type=activation)
    x = BatchNormalization(axis=1)(x)
    d_out = Dense(1, init='he_normal', activation='sigmoid')(x)

    return d_out, p_out, encoded


def GANAE(input_shape, order, batch_size=128, num_layers=4, train_disc=True):

    # Build Generator
    input_gen = Input(batch_shape=(batch_size,) + input_shape[:2] + (4,))
    g_out, l_dims = ToonGenAE(input_gen, num_layers=num_layers)
    generator = Model(input_gen, g_out)
    generator.name = make_name('ToonGenAE', num_layers=num_layers)

    # Build Encoder
    input_enc = Input(shape=input_shape)
    e_out, _ = ToonEncAE(input_enc, num_layers=num_layers)
    encoder = Model(input_enc, output=e_out)
    encoder.name = make_name('ToonEncAE', num_layers=num_layers)
    encoder.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))
    make_trainable(encoder, False)  # TODO: Best results? :O

    # Build decoder
    dec_in_shape = (batch_size, l_dims[-1], l_dims[-1], F_DIMS[num_layers])
    input_dec = Input(batch_shape=dec_in_shape)
    dec_out = ToonDecoder(input_dec, num_layers, l_dims, batch_size=batch_size)
    decoder = Model(input_dec, output=dec_out)
    decoder.name = make_name('ToonDecAE', num_layers=num_layers)

    # Build Discriminator
    input_disc = Input(batch_shape=dec_in_shape[:3] + (2*F_DIMS[num_layers],))
    d_out = discAE(input_disc)
    discriminator = Model(input_disc, d_out)

    # Build GAN
    if train_disc:
        generator = make_trainable(generator, False)
    else:
        discriminator = make_trainable(discriminator, False)
        encoder = make_trainable(encoder, False)
        decoder = make_trainable(decoder, False)

    gen_input = Input(batch_shape=(batch_size,) + input_shape[:2] + (4,))
    im_input = Input(batch_shape=(batch_size,) + input_shape)
    d_y = encoder(im_input)
    g_x = generator(gen_input)
    d_in = my_merge(g_x, d_y, order)
    d_out = discriminator(d_in)

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    if train_disc:
        dec_y = decoder(d_y)
        gan = Model(input=[gen_input, im_input], output=[d_out, dec_y])
        gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1.0, 5.0], optimizer=optimizer)
        gan.name = make_name('GANAEd_5', num_layers=num_layers)

    else:
        dec_x = decoder(g_x)
        gan = Model(input=[gen_input, im_input], output=[d_out, dec_x])
        gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1.0, 5.0], optimizer=optimizer)
        gan.name = make_name('GANAEg_5', num_layers=num_layers)

    return gan, generator, encoder, decoder, discriminator


def discAE(input_disc):
    x = Flatten()(input_disc)
    x = Dense(2048, init='he_normal')(x)
    x = Dropout(0.5)(x)
    x = my_activation(x, type='lrelu')
    x = BatchNormalization(axis=1)(x)
    x = Dense(2048, init='he_normal')(x)
    x = Dropout(0.5)(x)
    x = my_activation(x, type='lrelu')
    x = BatchNormalization(axis=1)(x)
    d_out = Dense(1, init='he_normal', activation='sigmoid')(x)
    return d_out


def AE(input_shape, num_layers=5, batch_size=128):
    # Build Encoder
    input_enc = Input(shape=input_shape)
    e_out, l_dims = ToonEncAE(input_enc, num_layers=num_layers)
    encoder = Model(input_enc, output=e_out)
    encoder.name = make_name('ToonEncAE', num_layers=num_layers)

    # Build decoder
    dec_in_shape = (batch_size, l_dims[-1], l_dims[-1], F_DIMS[num_layers])
    input_dec = Input(batch_shape=dec_in_shape)
    dec_out = ToonDecoder(input_dec, num_layers, l_dims, batch_size=batch_size)
    decoder = Model(input_dec, output=dec_out)
    decoder.name = make_name('ToonDecAE', num_layers=num_layers)

    input_im = Input(shape=input_shape)
    enc = encoder(input_im)
    rec = decoder(enc)
    AE = Model(input_im, rec)
    AE.compile('adam', 'mse')
    return AE, encoder, decoder


def max_val(y_true, y_pred):
    return -y_pred


def min_val_margin(y_true, y_pred):
    return K.maximum(-y_pred, 0.5)


def GAN(input_shape, order, batch_size=128, num_layers=4, load_weights=False, noise=None, train_disc=True):
    gen_in_shape = (batch_size,) + input_shape[:2] + (4,)

    # Build Generator
    input_gen = Input(batch_shape=gen_in_shape)
    # g_out, _ = ToonGen(input_gen, num_layers=num_layers, batch_size=batch_size)
    g_out, _ = ToonGenTransp(input_gen, num_layers=num_layers, batch_size=batch_size)
    generator = Model(input_gen, g_out)
    generator.name = make_name('ToonGen', num_layers=num_layers)

    # Build Discriminator
    input_disc = Input(shape=input_shape[:2] + (6,))
    d_out, de_out, _ = ToonDisc(input_disc, num_layers=num_layers, activation='lrelu')
    discriminator = Model(input_disc, output=[d_out, de_out])
    discriminator.name = make_name('ToonDisc', num_layers=num_layers)
    make_trainable(discriminator, False)

    # Load weights
    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Build GAN
    g_input = Input(batch_shape=gen_in_shape)
    im_input = Input(batch_shape=(batch_size,) + input_shape)
    g_x = generator(g_input)

    d_in = my_merge(g_x, im_input, order)
    d_out, de_out = discriminator(d_in)

    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    if train_disc:
        make_trainable(generator, False)
        gan = Model(input=[g_input, im_input], output=[d_out, de_out])
        gan.compile(loss=['binary_crossentropy'] * 2, load_weights=[1.0, 0.01], optimizer=optimizer)
        gan.name = make_name('ToonGANd_outter', num_layers=num_layers)

    else:
        make_trainable(discriminator, False)
        gan = Model(input=[g_input, im_input], output=[d_out, de_out, g_x])
        gan.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mse'], loss_weights=[1.0, 0.1, 50.0],
                    optimizer=optimizer)
        gan.name = make_name('ToonGANg_outter', num_layers=num_layers)

    return gan, generator, discriminator


def my_merge(a, b, order):
    def _my_merge(x, y, order):
        merged = K.switch(K.get_value(order) > 0,
                        merge([x, y], mode='concat'),
                        merge([y, x], mode='concat'))
        return merged

    def _my_merge_output_shape(input_shape):
        shape = list(input_shape)
        shape[-1] *= 2
        return tuple(shape)

    x = merge([a, b], mode=lambda x: _my_merge(x[0], x[1], order=order), output_shape=lambda x: _my_merge_output_shape(x[0]))

    return x


def Disc2(input_shape, load_weights=False, num_layers=4, noise=None):
    # Build the model
    input_d = Input(shape=input_shape[:2] + (6,))
    d_out, de_out, _ = ToonDisc(input_d, num_layers=num_layers, activation='relu', noise=noise)
    discriminator = Model(input_d, [d_out, de_out])

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss=['binary_crossentropy']*2, load_weights=[1.0, 0.05], optimizer=optimizer)
    return discriminator


def DiscAE(input_shape, load_weights=False, num_layers=4, noise=None):
    # Build the model
    input_gen = Input(shape=input_shape)

    d_out, d_dec, _ = ToonEncAE(input_gen, num_layers=num_layers, activation='relu', noise=noise)
    discriminator = Model(input_gen, [d_out, d_dec])
    discriminator.name = make_name('ToonDiscriminator', num_layers=num_layers)

    # Load weights
    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(discriminator.name)))

    # Compile
    optimizer = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[0.5, 20.0], optimizer=optimizer)
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


def my_activation(x, type='relu', alpha=0.2):
    if type == 'lrelu':
        return LeakyReLU(alpha=alpha)(x)
    else:
        return Activation(type)(x)


def sub(a, b):
    return merge([a, b], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])


def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred), axis=-1)


def l2_margin(y_true, y_pred):  # Idea: could pass in margin during training (similar to noise thingie)
    return -K.maximum(2 - K.mean(K.square(y_pred), axis=-1), 0)


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred), axis=-1)


def l1_margin(y_true, y_pred):  # Idea: could pass in margin during training (similar to noise thingie)
    return -K.maximum(0.0 - K.mean(K.abs(y_pred), axis=-1), 0)


def ld_0(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, K.zeros_like(y_pred)), axis=-1)


def ld_1(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, K.ones_like(y_pred)), axis=-1)


def Classifier(input_shape, batch_size=128, num_layers=4, num_classes=1000, net_load_name=None,
               compile_model=True, use_gen=False):
    # Build encoder
    if use_gen:
        input_im = Input(batch_shape=(batch_size,) + input_shape[:2] + (4,))
        decoded, encoded = ToonGen(input_im, num_layers=num_layers, batch_size=batch_size)
    else:
        input_im = Input(batch_shape=(batch_size,) + input_shape[:2] + (6,))
        decoded, _, encoded = ToonDisc(input_im, num_layers=num_layers)
    encoder = Model(input_im, encoded)
    discriminator = Model(input_im, decoded)
    if net_load_name:
        discriminator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_load_name)))
        make_trainable(encoder, False)

    # Build classifier
    if use_gen:
        im_input = Input(batch_shape=(batch_size,) + input_shape[:2] + (4,))
    else:
        im_input = Input(batch_shape=(batch_size,) + input_shape[:2] + (6,))
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
    return net


def disc_data(X, Y, Yd, p_wise=False, with_x=False):
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
