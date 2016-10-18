import tensorflow as tf
from keras.layers import Lambda, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge, Flatten, Dense, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential

NUM_CONV_LAYERS = 5
F_DIMS = [64, 128, 256, 512, 1024]
BN_MODE = 0


def ToonAE2(input_shape, batch_size, out_activation='tanh', num_res_layers=8, f_dims=(64, 128, 256, 512, 1024), bn_mode = 2):
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
    model = Sequential()
    l_dims = compute_layer_shapes(input_shape=input_shape)
    model.add(Lambda(lambda x: x, input_shape=input_shape))

    # Layer 1
    with tf.name_scope('conv_1'):
        model = conv_relu_bn(model, f_size=3, f_channels=32, stride=1, border='same', bn_mode=bn_mode)
        model = conv_relu_bn(model, f_size=4, f_channels=f_dims[0], stride=1, border='valid', bn_mode=bn_mode)

    # Layer 2
    with tf.name_scope('conv_2'):
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[1], stride=2, border='same', bn_mode=bn_mode)

    # Layer 3
    with tf.name_scope('conv_3'):
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[2], stride=2, border='valid', bn_mode=bn_mode)

    # Layer 4
    with tf.name_scope('conv_4'):
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[3], stride=2, border='valid', bn_mode=bn_mode)

    # Layer 5
    with tf.name_scope('conv_5'):
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[4], stride=2, border='valid', bn_mode=bn_mode)

    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            model.add(res_layer_bottleneck((l_dims[4], l_dims[4], f_dims[4]), f_dims[4], 256, bn_mode=bn_mode))

    # Layer 6
    with tf.name_scope('deconv_1'):
        model = upconv_bn(model, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        model.add(Activation('relu'))
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)

    # Layer 7
    with tf.name_scope('deconv_2'):
        model = upconv_bn(model, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        model.add(Activation('relu'))
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)

    # Layer 8
    with tf.name_scope('deconv_3'):
        model = upconv_bn(model, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        model.add(Activation('relu'))
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)

    # Layer 9
    with tf.name_scope('deconv_4'):
        model = upconv_bn(model, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same', bn_mode=bn_mode)
        model.add(Activation('relu'))
        model = conv_relu_bn(model, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)

    # Layer 10
    with tf.name_scope('deconv_5'):
        model = upconv_bn(model, f_size=4, f_channels=32, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid', bn_mode=bn_mode)
        model.add(Activation('relu'))
        model.add(Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal'))
        model.add(Activation(out_activation))

    # Create the model
    model.name = 'ToonAE2'

    return model


def ToonDiscriminator2(input_shape, num_res_layers=8, f_dims=[64, 128, 256, 512, 1024]):
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
    model.add(Lambda(lambda x: x, input_shape=input_shape))

    # Layer 1
    with tf.name_scope('conv_1'):
        model = conv_lrelu(model, f_size=3, f_channels=32, stride=1, border='valid')
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[0], stride=1, border='valid')

    # Layer 2
    with tf.name_scope('conv_2'):
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[0], stride=1, border='valid')
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[1], stride=2, border='valid')

    # Layer 3
    with tf.name_scope('conv_3'):
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[1], stride=1, border='valid')
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[2], stride=2, border='valid')

    # Layer 4
    with tf.name_scope('conv_4'):
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[2], stride=1, border='valid')
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[3], stride=2, border='valid')

    # Layer 5
    with tf.name_scope('conv_5'):
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[3], stride=1, border='valid')
        model = conv_lrelu_bn(model, f_size=3, f_channels=f_dims[4], stride=2, border='valid')

    # Res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            model.add(res_layer_bottleneck((8, 8, f_dims[4]), f_dims[4], 256))

    # Fully connected layer
    model = conv_lrelu_bn(model, f_size=1, f_channels=f_dims[3], stride=1, border='valid')
    model.add(Flatten())
    model.add(Dense(2048, init='he_normal'))
    model = lrelu(model)
    model.add(BatchNormalization(axis=1, mode=BN_MODE))
    model.add(Dense(1, init='he_normal'))
    model.add(Activation('sigmoid'))

    model.name = 'ToonDisc2'
    return model


def conv_relu_bn(model, f_size, f_channels, stride, border='valid', activation='relu', bn_mode=0):
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
    model.add(Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal'))
    model.add(Activation(activation))
    return model.add(BatchNormalization(axis=3, mode=bn_mode))


def conv_lrelu_bn(model, f_size, f_channels, stride, border='valid', bn_mode=0):
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
    model.add(Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal'))
    model = lrelu(model)
    return model.add(BatchNormalization(axis=3, mode=bn_mode))


def conv_relu(model, f_size, f_channels, stride, border='valid', activation='relu'):
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
    model.add(Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal'))
    return model.add(Activation(activation))


def conv_lrelu(model, f_size, f_channels, stride, border='valid'):
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
    model.add(Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal'))
    return lrelu(model)


def outter_connections(model, f_channels, bn_mode=0):
    """Wrapper for 1x1 convolutions used on the outer layers.

    Args:
        layer_in: Input to this layer
        f_channels: Number of channels for the output (usually the same as input)

    Returns:
        Result of convolution followed by batchnorm and leakyRelu
    """
    model.add(Convolution2D(f_channels, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal'))
    model.add(Activation('relu'))
    return model.add(BatchNormalization(axis=3, mode=bn_mode))


def upconv_bn(model, f_size, f_channels, out_dim, batch_size, stride, border='valid', bn_mode=0):
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
    model.add(Deconvolution2D(f_channels, f_size, f_size,
                        output_shape=(batch_size, out_dim, out_dim, f_channels),
                        border_mode=border,
                        subsample=(stride, stride),
                        init='he_normal'))
    return model.add(BatchNormalization(axis=3, mode=bn_mode))


def res_layer_bottleneck(input_shape, out_dim, bn_dim, activation='relu', bn_mode=0):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    l_in = Input(input_shape)
    # 1x1 Bottleneck
    x = conv_relu_bn(l_in, f_size=1, f_channels=bn_dim, stride=1, border='same', activation=activation, bn_mode=bn_mode)
    # 3x3 conv
    x = conv_relu_bn(x, f_size=3, f_channels=bn_dim, stride=1, border='same', activation=activation, bn_mode=bn_mode)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=bn_mode)(x)
    x = merge([x, l_in], mode='sum')
    x = Activation(activation)(x)
    block = Model(input=l_in, output=x)
    return block


def res_layer_bottleneck_lrelu(input_shape, out_dim, bn_dim, bn_mode=0):
    """Constructs a Residual-Layer with bottleneck 1x1 convolutions and 3x3 convolutions

    Args:
        in_layer: Input to residual-layer
        out_dim: Dimension (number of channels) of the output (should be the same as input)
        bn_dim: Dimension of the bottlenecked convolutions

    Returns:
        Output of same dimensionality as input
    """
    l_in = Input(input_shape)
    # 1x1 Bottleneck
    x = conv_lrelu_bn(l_in, f_size=1, f_channels=bn_dim, stride=1, border='same', bn_mode=bn_mode)
    # 3x3 conv
    x = conv_lrelu_bn(x, f_size=3, f_channels=bn_dim, stride=1, border='same', bn_mode=bn_mode)
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    x = BatchNormalization(axis=3, mode=bn_mode)(x)
    x = merge([x, l_in], mode='sum')
    x = LeakyReLU()(x)
    block = Model(input=l_in, output=x)
    return block


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


def lrelu(model, alpha=0.2):
    return model.add(LeakyReLU(alpha=alpha))


if __name__ == '__main__':
    in_dims = (256, 256, 3)
    print(compute_layer_shapes(in_dims))
    net = ToonAE2(in_dims, 250)
