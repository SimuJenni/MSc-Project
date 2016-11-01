import tensorflow as tf
import tensorflow.contrib.slim as slim

F_DIMS = [64, 96, 160, 256, 512]


def ToonGenerator(inputs, num_res=16):
    f_dims = F_DIMS
    with toon_net_argscope():
        # Conv1
        net = slim.conv2d(inputs, num_outputs=32, scope='conv_1', stride=1)
        net = slim.conv2d(net, num_outputs=f_dims[0], scope='conv_1', stride=2)

        # Conv2
        net = slim.conv2d(net, num_outputs=f_dims[0], scope='conv_2', stride=1)
        net = slim.conv2d(net, num_outputs=f_dims[1], scope='conv_2', stride=2)

        # Conv3
        net = slim.conv2d(net, num_outputs=f_dims[1], scope='conv_3', stride=1)
        net = slim.conv2d(net, num_outputs=f_dims[2], scope='conv_3', stride=2)

        # Conv4
        net = slim.conv2d(net, num_outputs=f_dims[2], scope='conv_4', stride=1)
        net = slim.conv2d(net, num_outputs=f_dims[3], scope='conv_4', stride=2)

        # Conv5
        net = slim.conv2d(net, num_outputs=f_dims[3], scope='conv_5', stride=1)
        net = slim.conv2d(net, num_outputs=f_dims[4], scope='conv_5', stride=2)

        # Res-Layers
        net = slim.repeat(net, num_res, res_layer_bottleneck, f_dims[4], f_dims[1], scope='res')

        # UpConv1
        net = up_conv2d(net, depth=f_dims[3], scope='upconv_1')
        net = slim.conv2d(net, num_outputs=f_dims[3], scope='upconv_1', stride=1)

        # UpConv2
        net = up_conv2d(net, depth=f_dims[2], scope='upconv_2')
        net = slim.conv2d(net, num_outputs=f_dims[2], scope='upconv_2', stride=1)

        # UpConv3
        net = up_conv2d(net, depth=f_dims[1], scope='upconv_3')
        net = slim.conv2d(net, num_outputs=f_dims[1], scope='upconv_3', stride=1)

        # UpConv4
        net = up_conv2d(net, depth=f_dims[0], scope='upconv_4')
        net = slim.conv2d(net, num_outputs=f_dims[0], scope='upconv_4', stride=1)

        # UpConv5
        net = up_conv2d(net, depth=32, scope='upconv_5')
        net = slim.conv2d(net, num_outputs=3, scope='upconv_5', stride=1, activation_fn=tf.nn.tanh)

        return net


def ToonDiscriminator(inputs, num_res=8):
    f_dims = F_DIMS
    with toon_net_argscope(activation=lrelu):
        # Conv1
        net = slim.conv2d(inputs, num_outputs=32, scope='conv_1', stride=1)
        l1 = slim.conv2d(net, num_outputs=f_dims[0], scope='conv_1', stride=2)

        # Conv2
        net = slim.conv2d(l1, num_outputs=f_dims[0], scope='conv_2', stride=1)
        l2 = slim.conv2d(net, num_outputs=f_dims[1], scope='conv_2', stride=2)

        # Conv3
        net = slim.conv2d(l2, num_outputs=f_dims[1], scope='conv_3', stride=1)
        l3 = slim.conv2d(net, num_outputs=f_dims[2], scope='conv_3', stride=2)

        # Conv4
        net = slim.conv2d(l3, num_outputs=f_dims[2], scope='conv_4', stride=1)
        l4 = slim.conv2d(net, num_outputs=f_dims[3], scope='conv_4', stride=2)

        # Conv5
        net = slim.conv2d(l4, num_outputs=f_dims[3], scope='conv_5', stride=1)
        l5 = slim.conv2d(net, num_outputs=f_dims[4], scope='conv_5', stride=2)

        # Res-Layers
        net = slim.repeat(l5, num_res, res_layer_bottleneck, f_dims[4], f_dims[1], scope='res')

        # Fully connected
        net = slim.fully_connected(net, 2048, scope='fc1')
        net = slim.fully_connected(net, 1, scope='fc2', activation_fn=tf.nn.sigmoid)

        return net, [l1, l2, l3, l4, l5]


def res_layer_bottleneck(inputs, depth, depth_bottleneck, scope=None):
    shortcut = inputs
    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='{}_conv1'.format(scope))
    residual = slim.conv2d(residual, depth_bottleneck, [3, 3], stride=1, padding='SAME', scope='{}_conv2'.format(scope))
    residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='{}_conv3'.format(scope))
    output = tf.nn.relu(shortcut + residual)
    return output


def up_conv2d(net, depth, scope, factor=2):
    in_shape = net.get_shape()
    net = tf.image.resize_images(net, factor * in_shape[1], factor * in_shape[2])  # TODO: Maybe use nearest neighbor?
    net = slim.conv2d(net, num_outputs=depth, scope=scope, stride=1)
    return net


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def toon_net_argscope(activation=tf.nn.relu):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope([slim.conv2d],
                        kernel_size=[3, 3],
                        activation_fn=activation,
                        padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc
