import os

import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge, Flatten, Dense, \
    UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from constants import MODEL_DIR

NUM_CONV_LAYERS = 5
# F_DIMS = [64, 128, 256, 512, 1024]
F_DIMS = [64, 96, 160, 256, 512]


def ToonAE(in_layer, input_shape, batch_size, out_activation='tanh', num_res_layers=8, f_dims=F_DIMS, bn_mode=0):
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
    l_dims = compute_layer_shapes(input_shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_relu_bn(in_layer, f_size=3, f_channels=32, stride=1, border='same', bn_mode=bn_mode)
        x = conv_relu_bn(x, f_size=4, f_channels=f_dims[0], stride=1, border='valid', bn_mode=bn_mode)

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', bn_mode=bn_mode)

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid', bn_mode=bn_mode)

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid', bn_mode=bn_mode)

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid', bn_mode=bn_mode)

    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[2], bn_mode=bn_mode)  # TODO: CHANGED BN_MODE

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        x = Activation('relu')(x)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        x = Activation('relu')(x)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid', bn_mode=bn_mode)
        x = Activation('relu')(x)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same', bn_mode=bn_mode)
        x = Activation('relu')(x)
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = upconv_bn(x, f_size=4, f_channels=32, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid',
                      bn_mode=bn_mode)
        x = Activation('relu')(x)
        x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
        decoded = Activation(out_activation)(x)

    return decoded


def ToonAEresize(in_layer, input_shape, out_activation='tanh', num_res_layers=8, f_dims=F_DIMS, bn_mode=0):
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
    l_dims = compute_layer_shapes(input_shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_relu(in_layer, f_size=3, f_channels=32, stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same', bn_mode=bn_mode)

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', bn_mode=bn_mode)

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same', bn_mode=bn_mode)

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same', bn_mode=bn_mode)

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same', bn_mode=bn_mode)

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[1], bn_mode=bn_mode, lightweight=True)

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=32, stride=1, border='same')
        x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
        decoded = Activation(out_activation)(x)

    return decoded


def ToonAEresizeWoutter(in_layer, input_shape, out_activation='tanh', num_res_layers=8, f_dims=F_DIMS, bn_mode=0):
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
    l_dims = compute_layer_shapes(input_shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_relu(in_layer, f_size=3, f_channels=32, stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=2, border='same', bn_mode=bn_mode)
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='same', bn_mode=bn_mode)
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='same', bn_mode=bn_mode)
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='same', bn_mode=bn_mode)
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='same', bn_mode=bn_mode)
        l5 = outter_connections(x, f_dims[4])

    # Residual layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[1], bn_mode=bn_mode, lightweight=True)
    x = merge([x, l5], mode='sum')

    # Layer 6
    with tf.name_scope('deconv_1'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[3], stride=1, border='same')
        x = merge([x, l4], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='same', bn_mode=bn_mode)

    # Layer 7
    with tf.name_scope('deconv_2'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[2], stride=1, border='same')
        x = merge([x, l3], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='same', bn_mode=bn_mode)

    # Layer 8
    with tf.name_scope('deconv_3'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[1], stride=1, border='same')
        x = merge([x, l2], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='same', bn_mode=bn_mode)

    # Layer 9
    with tf.name_scope('deconv_4'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=f_dims[0], stride=1, border='same')
        x = merge([x, l1], mode='sum')
        x = conv_relu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='same', bn_mode=bn_mode)

    # Layer 10
    with tf.name_scope('deconv_5'):
        x = UpSampling2D()(x)
        x = conv_relu(x, f_size=3, f_channels=32, stride=1, border='same')
        x = Convolution2D(3, 3, 3, border_mode='same', subsample=(1, 1), init='he_normal')(x)
        decoded = Activation(out_activation)(x)

    return decoded


def ToonDiscriminator(in_layer, num_res_layers=8, f_dims=F_DIMS, bn_mode=0):
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

    # Layer 1
    with tf.name_scope('conv_1'):
        x = conv_lrelu(in_layer, f_size=3, f_channels=32, stride=1, border='valid')
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='valid')

    # Layer 2
    with tf.name_scope('conv_2'):
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[0], stride=1, border='valid')
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[1], stride=2, border='valid')

    # Layer 3
    with tf.name_scope('conv_3'):
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[1], stride=1, border='valid')
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')

    # Layer 4
    with tf.name_scope('conv_4'):
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[2], stride=1, border='valid')
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')

    # Layer 5
    with tf.name_scope('conv_5'):
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[3], stride=1, border='valid')
        x = conv_lrelu_bn(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')

    # Res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck(x, f_dims[4], f_dims[2])

    # Fully connected layer
    x = conv_lrelu_bn(x, f_size=1, f_channels=f_dims[3], stride=1, border='valid')
    x = Flatten()(x)
    x = Dense(2048, init='he_normal')(x)
    x = lrelu(x)
    x = BatchNormalization(axis=1, mode=bn_mode)(x)
    x = Dense(1, init='he_normal')(x)
    x = Activation('sigmoid')(x)

    return x


def ToonDiscLwise(in_layer, num_res_layers=8, f_dims=F_DIMS, bn_mode=0):
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
        x = l5

    layer_activations = [l1, l2, l3, l4, l5]

    # Res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            x = res_layer_bottleneck_lrelu(x, f_dims[4], f_dims[1], lightweight=True)
            # layer_activations.append(x)   # TODO: Maybe remove again

    # Fully connected layer
    x = Flatten()(x)
    x = Dense(2048, init='he_normal')(x)
    x = lrelu(x)
    x = BatchNormalization(axis=1, mode=bn_mode)(x)
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


def outter_connections(layer_in, f_channels, bn_mode=0):
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


def upconv_bn(layer_in, f_size, f_channels, out_dim, batch_size, stride, border='valid', bn_mode=0):
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
    return BatchNormalization(axis=3, mode=bn_mode)(x)


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


def Generator(input_shape, batch_size, load_weights=False, f_dims=F_DIMS, resize_conv=False, w_outter=False):
    input_gen = Input(shape=input_shape)
    if resize_conv:
        if w_outter:
            gen_out = ToonAEresizeWoutter(input_gen, input_shape, f_dims=f_dims)
            net_name = 'ToonAEresizeWoutter'
        else:
            gen_out = ToonAEresize(input_gen, input_shape, f_dims=f_dims)
            net_name = 'ToonAEresize'
    else:
        gen_out = ToonAE(input_gen, input_shape, batch_size=batch_size, f_dims=f_dims)
        net_name = 'ToonAE'
    generator = Model(input_gen, gen_out)

    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mse', optimizer=optimizer)
    generator.name = net_name
    return generator


def Discriminator(input_shape, load_weights=False, f_dims=F_DIMS):
    input_disc = Input(shape=input_shape)
    dis_out = ToonDiscriminator(input_disc, f_dims=f_dims)
    discriminator = Model(input_disc, dis_out)

    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDisc.hdf5'))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    discriminator.name = 'ToonDisc'
    return discriminator


def DiscLwise(input_shape, load_weights=False, f_dims=F_DIMS, train=True):
    input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscLwise(input_disc, f_dims=f_dims)
    if train:
        discriminator = Model(input_disc, dis_out)
    else:
        discriminator = Model(input_disc, layer_activations + [dis_out])
        make_trainable(discriminator, False)

    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDiscLwise.hdf5'))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    discriminator.name = 'ToonDiscLwise'
    return discriminator


def DiscriminatorWithX(input_shape, load_weights=False, f_dims=F_DIMS):
    input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    dis_out = ToonDiscriminator(input_disc, f_dims=f_dims)
    discriminator = Model(input_disc, dis_out)

    if load_weights:
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDiscWithX.hdf5'))

    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator


def Gan(input_shape, batch_size, load_weights=False, f_dims=F_DIMS):
    input_gen = Input(shape=input_shape)
    gen_out = ToonAE(input_gen, input_shape=input_shape, batch_size=batch_size, f_dims=f_dims)
    generator = Model(input_gen, gen_out)

    input_disc = Input(shape=input_shape)
    dis_out = ToonDiscriminator(input_disc, f_dims=f_dims)
    discriminator = Model(input_disc, dis_out)
    make_trainable(discriminator, False)

    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, 'ToonAE.hdf5'))
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDisc.hdf5'))

    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    im_class = discriminator(im_recon)
    gan = Model(input=im_input, output=[im_class, im_recon])

    optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    reg = 25.0

    gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1.0, reg], optimizer=optimizer)
    return gan, generator, discriminator


def GanLwise(input_shape, batch_size, load_weights=False, f_dims=F_DIMS, resize_conv=False, w_outter=False,
             recon_weight=100.0):
    input_gen = Input(shape=input_shape)
    if resize_conv:
        if w_outter:
            gen_out = ToonAEresizeWoutter(input_gen, input_shape, f_dims=f_dims)
            net_name = 'ToonAEresizeWoutter'
        else:
            gen_out = ToonAEresize(input_gen, input_shape, f_dims=f_dims)
            net_name = 'ToonAEresize'
    else:
        gen_out = ToonAE(input_gen, input_shape, batch_size=batch_size, f_dims=f_dims)
    generator = Model(input_gen, gen_out)

    input_disc = Input(shape=input_shape)
    dis_out, layer_activations = ToonDiscLwise(input_disc, f_dims=f_dims)
    discriminator = Model(input_disc, output=layer_activations + [dis_out])
    make_trainable(discriminator, False)

    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDiscLwise.hdf5'))

    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    disc_out = discriminator(im_recon)
    gan = Model(input=im_input, output=disc_out + [im_recon])

    optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    disc_weight = 1.0
    layer_weights = 0.1

    gan.compile(loss=['mse'] * len(layer_activations) + ['binary_crossentropy', 'mse'],
                loss_weights=[layer_weights * (i + 1) for i in range(len(layer_activations))] + [disc_weight,
                                                                                                 recon_weight],
                optimizer=optimizer)
    return gan, generator, discriminator


def GanWithX(input_shape, batch_size, load_weights=False, f_dims=F_DIMS, l2_rate=25.0):
    input_gen = Input(shape=input_shape)
    gen_out = ToonAE(input_gen, input_shape=input_shape, batch_size=batch_size, f_dims=f_dims)
    generator = Model(input_gen, gen_out)

    input_disc = Input(shape=input_shape[:2] + (input_shape[2] * 2,))
    dis_out = ToonDiscriminator(input_disc, f_dims=f_dims)
    discriminator = Model(input_disc, dis_out)
    make_trainable(discriminator, False)

    if load_weights:
        generator.load_weights(os.path.join(MODEL_DIR, 'ToonAE.hdf5'))
        discriminator.load_weights(os.path.join(MODEL_DIR, 'ToonDiscWithX.hdf5'))

    im_input = Input(shape=input_shape)
    im_recon = generator(im_input)
    disc_in = merge([im_input, im_recon], mode='concat')
    im_class = discriminator(disc_in)
    gan = Model(input=im_input, output=[im_class, im_recon])

    optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    gan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1.0, l2_rate], optimizer=optimizer)
    return gan, generator, discriminator


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


if __name__ == '__main__':
    in_dims = (256, 256, 3)
    print(compute_layer_shapes(in_dims))
    net = ToonAE(in_dims, 250)
