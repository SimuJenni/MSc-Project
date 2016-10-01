from keras.layers import Input, Convolution2D, BatchNormalization, Deconvolution2D, Activation, merge
from keras.models import Model

NUM_CONV_LAYERS = 5
F_DIMS = [64, 128, 256, 512, 1024]

def ToonNet(input_shape, batch_size):

    # Compute the dimensions of the layers
    l_dims = compute_layer_dims(input_shape=input_shape)
    input_im = Input(shape=input_shape)

    # Layer 1
    x = Convolution2D(F_DIMS[0], 4, 4, border_mode='valid', subsample=(1, 1))(input_im)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 2
    x = Convolution2D(F_DIMS[1], 3, 3, border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 3
    x = Convolution2D(F_DIMS[2], 3, 3, border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 4
    x = Convolution2D(F_DIMS[3], 3, 3, border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 5
    x = Convolution2D(F_DIMS[4], 3, 3, border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    encoded = Activation('relu')(x)

    # Layer 6
    x = Deconvolution2D(F_DIMS[3], 3, 3, output_shape=(batch_size, l_dims[4], l_dims[4], F_DIMS[3]), border_mode='valid',
                        subsample=(2, 2))(encoded)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 7
    x = Deconvolution2D(F_DIMS[2], 3, 3, output_shape=(batch_size, l_dims[3], l_dims[3], F_DIMS[2]), border_mode='valid',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 8
    x = Deconvolution2D(F_DIMS[1], 3, 3, output_shape=(batch_size, l_dims[2], l_dims[2], F_DIMS[1]), border_mode='valid',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 9
    x = Deconvolution2D(F_DIMS[0], 3, 3, output_shape=(batch_size, l_dims[1], l_dims[1], F_DIMS[0]), border_mode='same',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Layer 10
    x = Deconvolution2D(3, 4, 4, output_shape=(batch_size, l_dims[0], l_dims[0], 3), border_mode='valid',
                        subsample=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    decoded = Activation('tanh')(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.summary()
    toon_net.compile(optimizer='adam', loss='mse')

    return toon_net, encoded

def ToonResNet(input_shape, batch_size):

    # Compute the dimensions of the layers
    l_dims = compute_layer_dims(input_shape=input_shape)
    input_im = Input(shape=input_shape)

    # Layer 1
    x = Convolution2D(F_DIMS[0], 4, 4, border_mode='valid', subsample=(1, 1))(input_im)
    x = BatchNormalization(axis=3)(x)
    l1 = Activation('relu')(x)

    # Layer 2
    x = Convolution2D(F_DIMS[1], 3, 3, border_mode='same', subsample=(2, 2))(l1)
    x = BatchNormalization(axis=3)(x)
    l2 = Activation('relu')(x)

    # Layer 3
    x = Convolution2D(F_DIMS[2], 3, 3, border_mode='valid', subsample=(2, 2))(l2)
    x = BatchNormalization(axis=3)(x)
    l3 = Activation('relu')(x)

    # Layer 4
    x = Convolution2D(F_DIMS[3], 3, 3, border_mode='valid', subsample=(2, 2))(l3)
    x = BatchNormalization(axis=3)(x)
    l4 = Activation('relu')(x)

    # Layer 5
    x = Convolution2D(F_DIMS[4], 3, 3, border_mode='valid', subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    encoded = Activation('relu')(x)

    # Layer 6
    x = Deconvolution2D(F_DIMS[3], 3, 3, output_shape=(batch_size, l_dims[4], l_dims[4], F_DIMS[3]), border_mode='valid',
                        subsample=(2, 2))(encoded)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, l4], mode='sum')
    x = Activation('relu')(x)

    # Layer 7
    x = Deconvolution2D(F_DIMS[2], 3, 3, output_shape=(batch_size, l_dims[3], l_dims[3], F_DIMS[2]), border_mode='valid',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, l3], mode='sum')
    x = Activation('relu')(x)

    # Layer 8
    x = Deconvolution2D(F_DIMS[1], 3, 3, output_shape=(batch_size, l_dims[2], l_dims[2], F_DIMS[1]), border_mode='valid',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, l2], mode='sum')
    x = Activation('relu')(x)

    # Layer 9
    x = Deconvolution2D(F_DIMS[0], 3, 3, output_shape=(batch_size, l_dims[1], l_dims[1], F_DIMS[0]), border_mode='same',
                        subsample=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = merge([x, l1], mode='sum')
    x = Activation('relu')(x)

    # Layer 10
    x = Deconvolution2D(3, 4, 4, output_shape=(batch_size, l_dims[0], l_dims[0], 3), border_mode='valid',
                        subsample=(1, 1))(x)
    x = BatchNormalization(axis=3)(x)
    decoded = Activation('tanh')(x)

    # Create the model
    toon_net = Model(input_im, decoded)
    toon_net.summary()
    toon_net.compile(optimizer='adam', loss='mse')

    return toon_net, encoded


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
    net = ToonResNet(in_dims, 250)
