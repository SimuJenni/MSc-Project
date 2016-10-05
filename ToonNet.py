from keras.layers import Input, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge
from keras.models import Model
import tensorflow as tf

NUM_CONV_LAYERS = 5
# F_DIMS = [32, 64, 128, 256, 512]
# F_DIMS = [64, 96, 128, 256, 512]
F_DIMS = [48, 80, 128, 208, 336]


def ToonNet(input_shape, batch_size, out_activation='sigmoid', num_res_layers=10):
    """Constructs a fully convolutional residual auto-encoder network.
    The network has the follow architecture:

    Layer           Filters     Stride  Connected

    L1: Conv-layer  4x4x64      1       L25         =======================================
    L2: Conv-layer  3x3x96      2       L24               ===========================
    L3: Conv-layer  3x3x128     2       L23                    =================
    L4: Conv-layer  3x3x256     2       L22                        =========
    L5: Conv-layer  3x3x512     2                                     ===
                                                                      |_|
    L6:                                                               |_|
    .               1x1x64      1                                     |_|
    .   Res-Layers  3x3x64      1                                     |_|
    .               3x3x512     1                                     |_|
    L20:                                                              |_|
                                                                      |_|
    L21: UpConv     3x3x256     2                                     ===
    L22: UpConv     3x3x128     2       L4                         =========
    L23: UpConv     3x3x96      2       L3                     =================
    L24: UpConv     3x3x64      2       L2                ===========================
    L25: UpConv     4x4x64      1       L1          =======================================

    Args:
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
        x = conv_bn_relu(input_im, f_size=4, f_channels=F_DIMS[0], stride=1, border='valid')
        l1 = Convolution2D(F_DIMS[0], 1, 1, border_mode='valid', subsample=(1, 1))(x)
        l1 = BatchNormalization(axis=3)(l1)

    # Layer 2
    with tf.name_scope('conv_layer_2'):
        x = conv_bn_relu(x, f_size=3, f_channels=F_DIMS[1], stride=2, border='same')
        l2 = Convolution2D(F_DIMS[1], 1, 1, border_mode='valid', subsample=(1, 1))(x)
        l2 = BatchNormalization(axis=3)(l2)

    # Layer 3
    with tf.name_scope('conv_layer_3'):
        x = conv_bn_relu(x, f_size=3, f_channels=F_DIMS[2], stride=2, border='valid')
        l3 = Convolution2D(F_DIMS[2], 1, 1, border_mode='valid', subsample=(1, 1))(x)
        l3 = BatchNormalization(axis=3)(l3)

    # Layer 4
    with tf.name_scope('conv_layer_4'):
        x = conv_bn_relu(x, f_size=3, f_channels=F_DIMS[3], stride=2, border='valid')
        l4 = Convolution2D(F_DIMS[3], 1, 1, border_mode='valid', subsample=(1, 1))(x)
        l4 = BatchNormalization(axis=3)(l4)

    # Layer 5
    with tf.name_scope('conv_layer_5'):
        encoded = conv_bn_relu(x, f_size=3, f_channels=F_DIMS[4], stride=2, border='valid')

    # All the res-layers
    for i in range(num_res_layers):
        with tf.name_scope('res_layer_{}'.format(i+1)):
            encoded = res_layer_bottleneck(encoded, F_DIMS[4], 64)

    # Layer 6
    with tf.name_scope('deconv_layer_1'):
        x = upconv_bn(encoded, f_size=3, f_channels=F_DIMS[3], out_dim=l_dims[4], batch_size=batch_size, stride=2,
                      border='valid')
        x = merge([x, l4], mode='concat')
        x = Activation('relu')(x)

    # Layer 7
    with tf.name_scope('deconv_layer_2'):
        x = upconv_bn(x, f_size=3, f_channels=F_DIMS[2], out_dim=l_dims[3], batch_size=batch_size, stride=2, border='valid')
        x = merge([x, l3], mode='concat')
        x = Activation('relu')(x)

    # Layer 8
    with tf.name_scope('deconv_layer_3'):
        x = upconv_bn(x, f_size=3, f_channels=F_DIMS[1], out_dim=l_dims[2], batch_size=batch_size, stride=2, border='valid')
        x = merge([x, l2], mode='concat')
        x = Activation('relu')(x)

    # Layer 9
    with tf.name_scope('deconv_layer_4'):
        x = upconv_bn(x, f_size=3, f_channels=F_DIMS[0], out_dim=l_dims[1], batch_size=batch_size, stride=2, border='same')
        x = merge([x, l1], mode='concat')
        x = Activation('relu')(x)

    # Layer 10
    with tf.name_scope('deconv_layer_5'):
        x = upconv_bn(x, f_size=4, f_channels=3, out_dim=l_dims[0], batch_size=batch_size, stride=1, border='valid')
        decoded = Activation(out_activation)(x)

    # Create the model
    toon_net = Model(input_im, decoded)

    return toon_net, encoded, decoded


def conv_bn_relu(layer_in, f_size, f_channels, stride, border='valid'):
    x = Convolution2D(f_channels, f_size, f_size, border_mode=border, subsample=(stride, stride))(layer_in)
    x = BatchNormalization(axis=3)(x)
    return Activation('relu')(x)


def upconv_bn(layer_in, f_size, f_channels, out_dim, batch_size, stride, border='valid'):
    x = Deconvolution2D(f_channels, f_size, f_size, output_shape=(batch_size, out_dim, out_dim, f_channels),
                        border_mode=border,
                        subsample=(stride, stride))(layer_in)
    return BatchNormalization(axis=3)(x)


def res_layer_bottleneck(in_layer, out_dim, bn_dim):
    # 1x1 Bottleneck
    x = conv_bn_relu(in_layer, f_size=1, f_channels=bn_dim, stride=1, border='same')
    # 3x3 conv
    x = conv_bn_relu(x, f_size=3, f_channels=bn_dim, stride=1, border='same')
    # 1x1 to out_dim
    x = Convolution2D(out_dim, 1, 1, border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, in_layer], mode='sum')
    x = Activation('relu')(x)
    return x


def compute_layer_dims(input_shape):
    layer_dims = [None] * (NUM_CONV_LAYERS + 1)
    dim = input_shape[0]

    layer_dims[0] = dim
    layer_dims[1] = dim - 3
    for i in range(2, NUM_CONV_LAYERS + 1):
        dim = (dim - 3) // 2 + 1
        layer_dims[i] = dim

    return layer_dims


if __name__ == '__main__':
    in_dims = (256, 256, 3)
    print(compute_layer_dims(in_dims))
    net = ToonNet(in_dims, 250)
