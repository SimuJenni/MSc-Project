import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

NUM_CONV_LAYERS = 5
# F_DIMS = [64, 96, 128, 256, 512]
F_DIMS = [64, 96, 160, 256, 416, 512]
#F_DIMS = [48, 80, 128, 208, 336]


def ToonNet(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10, merge_mode='concat', f_dims=F_DIMS):
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
        l_dims = compute_layer_dims(input_shape=input_shape)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_layer_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            encoded = res_layer_bottleneck(encoded, f_dims[4], 64)

    # Layer 6
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l3], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 8
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l2], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = merge([x, l1], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.name = 'ToonNet'

    return toon_net, encoded, decoded


def ToonNetEvenMore(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10, merge_mode='concat', f_dims=F_DIMS):
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
        l_dims = compute_layer_dims(input_shape=input_shape)
        input_im = Input(shape=input_shape)
        l0 = outter_connections(input_im, 3)

    # Layer 1
    with tf.name_scope('conv_layer_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')
        l5 = outter_connections(encoded, f_dims[4])

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            encoded = res_layer_bottleneck(encoded, f_dims[4], 64)
    encoded = merge([encoded, l5], mode=merge_mode)
    encoded = Activation('relu')(encoded)

    # Layer 6
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l3], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 8
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l2], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = merge([x, l1], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        x = merge([x, l0], mode=merge_mode)
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.name = 'ToonNetEvenMore'

    return toon_net, encoded, decoded


def ToonNetDeep(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10, merge_mode='concat', f_dims=F_DIMS):
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
        l_dims = compute_layer_dims(input_shape=input_shape, num_conv=6)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_layer_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')
        l5 = outter_connections(x, f_dims[4])

    # Layer 6
    with tf.name_scope('conv_layer_6'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=f_dims[5], stride=2, border='valid')
        l6 = outter_connections(encoded, f_dims[5])

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            encoded = res_layer_bottleneck(encoded, f_dims[5], 64)
    encoded = merge([encoded, l6], mode=merge_mode)
    encoded = Activation('relu')(encoded)

    # Up-Layer 1
    with tf.name_scope('conv_layer_6'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[4], out_dim=l_dims[5], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l5], mode=merge_mode)
        x = Activation('relu')(x)

    # Up-Layer 2
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode=merge_mode)
        x = Activation('relu')(x)

    # Up-Layer 3
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l3], mode=merge_mode)
        x = Activation('relu')(x)

    # Up-Layer 4
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l2], mode=merge_mode)
        x = Activation('relu')(x)

    # Up-Layer 5
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = merge([x, l1], mode=merge_mode)
        x = Activation('relu')(x)

    # Up-Layer 6
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.name = 'ToonNetMore'

    return toon_net, encoded, decoded



def ToonNetMore(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10, merge_mode='concat', f_dims=F_DIMS):
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
        l_dims = compute_layer_dims(input_shape=input_shape)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_layer_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')
        l1 = outter_connections(x, f_dims[0])

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')
        l2 = outter_connections(x, f_dims[1])

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')
        l3 = outter_connections(x, f_dims[2])

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')
        l4 = outter_connections(x, f_dims[3])

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')
        l5 = outter_connections(encoded, f_dims[4])

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            encoded = res_layer_bottleneck(encoded, f_dims[4], 64)
    encoded = merge([encoded, l5], mode=merge_mode)
    encoded = Activation('relu')(encoded)

    # Layer 6
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l3], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 8
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l2], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = merge([x, l1], mode=merge_mode)
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.name = 'ToonNetMore'

    return toon_net, encoded, decoded


def ToonNetNoOutter(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10, merge_mode='concat', f_dims=F_DIMS):
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
        l_dims = compute_layer_dims(input_shape=input_shape)
        input_im = Input(shape=input_shape)

    # Layer 1
    with tf.name_scope('conv_layer_1'):
        x = conv_bn_relu(input_im, f_size=4, f_channels=f_dims[0], stride=1, border='valid')

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[1], stride=2, border='same')

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[2], stride=2, border='valid')

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=f_dims[3], stride=2, border='valid')

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=f_dims[4], stride=2, border='valid')

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i + 1)):
            encoded = res_layer_bottleneck(encoded, f_dims[4], 64)

    # Layer 6
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=f_dims[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[2], out_dim=l_dims[3], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)

    # Layer 8
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[1], out_dim=l_dims[2], batch_size=batch_size, stride=2,
                      border='valid')
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=f_dims[0], out_dim=l_dims[1], batch_size=batch_size, stride=2,
                      border='same')
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.name = 'ToonNetNoOutter'

    return toon_net, encoded, decoded


def conv_bn_relu(layer_in, f_size, f_channels, stride, border='valid'):
    x = Convolution2D(f_channels, f_size, f_size,
                      border_mode=border,
                      subsample=(stride, stride),
                      init='he_normal')(layer_in)
    x = BatchNormalization(axis=3)(x)
    return Activation('relu')(x)


def outter_connections(layer_in, f_channels):
    l = Convolution2D(f_channels, 1, 1, border_mode='valid', subsample=(1, 1), init='he_normal')(layer_in)
    l = BatchNormalization(axis=3)(l)
    return LeakyReLU(alpha=0.3)(l)


def upconv_bn(layer_in, f_size, f_channels, out_dim, batch_size, stride, border='valid'):
    x = Deconvolution2D(f_channels, f_size, f_size,
                        output_shape=(batch_size, out_dim, out_dim, f_channels),
                        border_mode=border,
                        subsample=(stride, stride),
                        init='he_normal')(layer_in)
    return BatchNormalization(axis=3)(x)


def res_layer_bottleneck(in_layer, out_dim, bn_dim):
    # 1x1 Bottleneck
    x = conv_bn_relu(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same')
    # 3x3 conv
    x = conv_bn_relu(x, f_size=3, f_channels=bn_dim, stride=1, border='same')
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1), init='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, in_layer], mode='sum')
    x = Activation('relu')(x)
    return x


def compute_layer_dims(input_shape, num_conv=NUM_CONV_LAYERS):
    layer_dims = [None] * (num_conv + 1)
    dim = input_shape[0]

    layer_dims[0] = dim
    layer_dims[1] = dim - 3
    for i in range(2, num_conv + 1):
        dim = (dim - 3) // 2 + 1
        layer_dims[i] = dim

    return layer_dims


if __name__ == '__main__':
    in_dims = (256, 256, 3)
    print(compute_layer_dims(in_dims))
    net = ToonNet(in_dims, 250)
