import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge, Flatten, Dense, Dropout
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU

NUM_CONV_LAYERS = 5
F_DIMS = [64, 96, 160, 256, 512]


def ToonAE(input_shape, batch_size, out_activation='tanh', num_res_layers=8, merge_mode='sum', f_dims=F_DIMS):
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
        input_shape: Shape of the input images (height, width, channels)
        batch_size: Number of images per batch
        out_activation: Type of activation for last layer ('relu', 'sigmoid', 'tanh', ...)
        num_res_layers: Number of residual layers in the middle

    Returns:
        (net, encoded): The resulting Keras model (net) and the encoding layer
    """
    # Compute the dimensions of the layers
    with tf.name_scope('input'):
        l_dims = compute_layer_shapes(input_shape=input_shape)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')
        l5 = outter_connections(x, f_dims[4])

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], 64)
    x = merge([x, l5], mode=merge_mode)
    encoded = Activation('relu')(x)

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l3], mode=merge_mode)
        x = lrelu(x)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l2], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = merge([x, l1], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    model = Model(input_im, decoded)
    model.name = 'ToonAE'

    return model


def ToonAE2(input_shape, batch_size, out_activation='tanh', num_res_layers=4, merge_mode='sum', f_dims=F_DIMS):
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
        input_shape: Shape of the input images (height, width, channels)
        batch_size: Number of images per batch
        out_activation: Type of activation for last layer ('relu', 'sigmoid', 'tanh', ...)
        num_res_layers: Number of residual layers in the middle

    Returns:
        (net, encoded): The resulting Keras model (net) and the encoding layer
    """
    # Compute the dimensions of the layers
    with tf.name_scope('input'):
        l_dims = compute_layer_shapes(input_shape=input_shape)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_relu(input_im, f_size=3, f_channels=32, stride=1, border='same')
        x = conv_bn_relu(x, f_size=4, f_channels=f_dims[0], stride=1, border='valid')

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')

    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], 256)

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = Activation('relu')(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = upconv_bn(x, f_size=4, f_channels=32, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        x = Activation(out_activation)(x)
        decoded = conv_relu(x, f_size=3, f_channels=3, stride=1, border='same')

    # Create the model
    model = Model(input_im, decoded)
    model.name = 'ToonAE'

    return model


def ToonDiscriminator(input_shape):
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
    model = Sequential()
    model.name = 'ToonDisc'

    # Conv-Layer 1
    model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal', input_shape=input_shape))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU(alpha=0.2))

    # Conv-Layer 2
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU(alpha=0.2))

    # Conv-Layer 3
    model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU(alpha=0.2))

    # Conv-Layer 4
    model.add(Convolution2D(256, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU(alpha=0.2))

    # Conv-Layer 5
    model.add(Convolution2D(512, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU(alpha=0.2))

    # Fully connected layer 1
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))

    # Fully connected layer 2
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))

    # Fully connected layer 3
    model.add(Dense(2, activation='softmax'))
    return model

def ToonDiscriminator3(input_shape):
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
    model = Sequential()
    model.name = 'ToonDisc3'

    # Conv-Layer 1
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), input_shape=input_shape))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())

    # Conv-Layer 2
    model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())

    # Conv-Layer 3
    model.add(Convolution2D(256, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())
    model.add(Convolution2D(256, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())

    # Conv-Layer 4
    model.add(Convolution2D(512, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())
    model.add(Convolution2D(512, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())

    # Conv-Layer 5
    model.add(Convolution2D(1024, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())
    model.add(Convolution2D(1024, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(BatchNormalization(axis=3, mode=2))
    model.add(LeakyReLU())

    # Fully connected layer 1
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    # Fully connected layer 3
    model.add(Dense(2, activation='softmax'))
    return model


def ToonDiscriminator2(input_shape, num_res_layers=16):
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
    input_im = Input(shape=input_shape)

    # Conv-Layer 1
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal')(input_im)
    x = BatchNormalization(axis=3, mode=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Conv-Layer 2
    x = Convolution2D(128, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Conv-Layer 3
    x = Convolution2D(256, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Conv-Layer 4
    x = Convolution2D(512, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Conv-Layer 5
    x = Convolution2D(1024, 3, 3, border_mode='valid', subsample=(2, 2), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # All the res-layers
    for i in range(num_res_layers):
        x = res_layer_bottleneck(x, 1024, 64, LeakyReLU(alpha=0.2))

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(2048, init='he_normal')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    pred = Dense(1, activation='sigmoid', init='he_normal')(x)

    model = Model(input_im, pred)
    model.name = 'ToonDist2'
    return model


def conv_bn_relu(layer_in, f_size, f_channels, stride, border='valid', activation='relu'):
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
    x = BatchNormalization(axis=3)(x)
    return Activation(activation)(x)


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


def outter_connections(layer_in, f_channels):
    """Wrapper for 1x1 convolutions used on the outer layers.

    Args:
        layer_in: Input to this layer
        f_channels: Number of channels for the output (usually the same as input)

    Returns:
        Result of convolution followed by batchnorm and leakyRelu
    """
    l = Convolution2D(f_channels, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal')(layer_in)
    l = BatchNormalization(axis=3)(l)
    return Activation('relu')(l)


def upconv_bn(layer_in, f_size, f_channels, out_dim, batch_size, stride, border='valid'):
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
    return BatchNormalization(axis=3)(x)


def res_layer_bottleneck(in_layer, out_dim, bn_dim, activation='relu'):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    # 1x1 Bottleneck
    x = conv_bn_relu(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation)
    # 3x3 conv
    x = conv_bn_relu(x, f_size=3, f_channels=bn_dim, stride=1, border='same', activation=activation)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, in_layer], mode='sum')
    return Activation(activation)(x)


def compute_layer_shapes(input_shape, num_conv=NUM_CONV_LAYERS):
    """Helper function computing the shapes of the output layers resulting from strided convolutions.

    Args:
        input_shape: Shape of the input (expecting square input images)
        num_conv: Number of strided convolution layers (not counting res-layers)

    Returns:
        List of resulting layer-dimensions
    """
    layer_dims = [None] * (num_conv + 1)
    dim = input_shape[0]

    layer_dims[0] = dim
    layer_dims[1] = dim - 3
    for i in range(2, num_conv + 1):
        dim = (dim - 3) // 2 + 1
        layer_dims[i] = dim
    return layer_dims


def lrelu(x, alpha=0.2):
    return LeakyReLU(alpha=alpha)(x)


if __name__ == '__main__':
    in_dims = (256, 256, 3)
    print(compute_layer_shapes(in_dims))
    net = ToonAE(in_dims, 250)
