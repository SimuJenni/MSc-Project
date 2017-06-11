import pickle
import os

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


class AlexNetConverter:
    def __init__(self, model_dir, model, sess, remove_bn=False, ckpt=None, net_id='discriminator', scale=1.0, bgr=False,
                 exclude=None, with_fc=False, pad='VALID', im_size=(224, 224), use_classifier=True, num_classes=1000):
        self.model = model
        self.bgr = bgr
        self.sess = sess
        self.net_id = net_id
        self.bn_eps = 0.001
        self.weights_file = '{}/weights_dict'.format(model_dir)
        self.remove_bn = remove_bn
        self.scale = scale
        self.exclude = exclude
        self.with_fc = with_fc
        self.pad = pad
        self.im_size = im_size
        self.use_classifier = use_classifier
        self.num_classes = num_classes
        if ckpt:
            self.ckpt = ckpt
        else:
            self.ckpt = tf.train.get_checkpoint_state(model_dir)

    def init_model(self):
        x = tf.Variable(tf.random_normal([1, self.im_size[0], self.im_size[1], 3], stddev=2, seed=42), name='x')
        if self.use_classifier:
            self.model.build_classifier(x, self.num_classes)
        else:
            self.model.discriminator.discriminate(x, with_fc=self.with_fc, training=False, pad=self.pad)
        self.sess.run(tf.global_variables_initializer())
        var2restore = slim.get_variables_to_restore(include=[self.net_id], exclude=self.exclude)
        if self.use_classifier:
            var2restore += slim.get_variables_to_restore(include=['fully_connected'])
        print('Variables: {}'.format([v.op.name for v in var2restore]))
        saver = tf.train.Saver(var_list=var2restore)
        saver.restore(self.sess, self.ckpt)

    def get_bn_params(self, layer_id, fc_scope=None):
        print('Extracting {}/BatchNorm'.format(layer_id))
        if fc_scope:
            scope = fc_scope
        else:
            scope = self.net_id
        with tf.variable_scope(scope, reuse=True):
            beta = slim.variable('{}/BatchNorm/beta'.format(layer_id))
            moving_mean = slim.variable('{}/BatchNorm/moving_mean'.format(layer_id))
            moving_variance = slim.variable('{}/BatchNorm/moving_variance'.format(layer_id))
            return moving_mean, moving_variance, beta

    def get_conv_weights(self, layer, scope='conv_{}/weights'):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable(scope.format(layer))

    def get_conv_biases(self, layer, scope='conv_{}/biases'):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable(scope.format(layer))

    def get_fc_weights(self, layer=None):
        scope = 'fully_connected' if self.use_classifier else self.net_id
        with tf.variable_scope(scope, reuse=True):
            if layer:
                return slim.variable('fc{}/weights'.format(layer))
            else:
                return slim.variable('fc3/weights') if self.use_classifier else slim.variable('fully_connected/weights')

    def get_fc_biases(self, layer=None):
        scope = 'fully_connected' if self.use_classifier else self.net_id
        with tf.variable_scope(scope, reuse=True):
            if layer:
                return slim.variable('fc{}/biases'.format(layer))
            else:
                return slim.variable('fc3/biases') if self.use_classifier else slim.variable('fully_connected/biases')

    def get_conv(self, layer):
        weights = self.get_conv_weights(layer)
        biases = self.get_conv_biases(layer)
        return weights, biases

    def get_conv_rm_bn(self, layer, scope='conv_{}'):
        weights = self.get_conv_weights(layer, scope='{}/weights'.format(scope))
        moving_mean, moving_variance, beta = self.get_bn_params(scope.format(layer))
        alpha = tf.rsqrt(moving_variance + self.bn_eps)
        weights *= alpha
        biases = beta - moving_mean * alpha
        return weights, biases

    def get_group_conv_rm_bn(self, layer):
        weights_0, biases_0 = self.get_conv_rm_bn(layer, scope='conv_{}_0')
        weights_1, biases_1 = self.get_conv_rm_bn(layer, scope='conv_{}_1')
        weights = tf.concat(3, [weights_0, weights_1])
        biases = tf.concat(0, [biases_0, biases_1])
        return weights, biases

    def get_fc(self, layer=None):
        weights = self.get_fc_weights(layer)
        biases = self.get_fc_biases(layer)
        return weights, biases

    def get_fc_rm_bn(self, layer):
        weights = self.get_fc_weights(layer)
        scope = 'fully_connected' if self.use_classifier else None
        moving_mean, moving_variance, beta = self.get_bn_params('fc{}'.format(layer), fc_scope=scope)
        alpha = tf.rsqrt(moving_variance + self.bn_eps)
        weights *= alpha
        biases = beta - moving_mean * alpha
        return weights, biases

    def hwcn2nchw(self, input):
        return np.transpose(input, [3, 2, 0, 1])

    def nchw2hwcn(self, input):
        return np.transpose(input, [2, 3, 1, 0])

    def extract_and_store(self):
        self.init_model()
        self.extract_and_store_remove_batchnorm()

    def extract_and_store_remove_batchnorm(self):
        num_conv = self.model.num_layers
        weights_dict = {}
        for l in range(num_conv):
            if l == 0:
                weights, biases = self.get_conv(l+1)
            elif l == 2:
                weights, biases = self.get_conv_rm_bn(l+1)
            else:
                weights, biases = self.get_group_conv_rm_bn(l + 1)
            weights_dict['conv_{}/weights'.format(l+1)] = weights.eval()
            weights_dict['conv_{}/biases'.format(l+1)] = biases.eval()
        if self.with_fc:
            for l in range(2):
                weights, biases = self.get_fc_rm_bn(l+1)
                weights_dict['fc{}/weights'.format(l+1)] = weights.eval()
                weights_dict['fc{}/biases'.format(l+1)] = biases.eval()
            weights, biases = self.get_fc()
            weights_dict['fully_connected/weights'] = weights.eval()
            weights_dict['fully_connected/biases'] = biases.eval()
        self.save_weights(weights_dict)

    def save_weights(self, weights_dict):
        with open(self.weights_file + '.pkl', 'wb') as f:
            pickle.dump(weights_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self):
        with open(self.weights_file + '.pkl', 'rb') as f:
            return pickle.load(f)

    def load_and_set_caffe_weights(self, proto_path, save_path):
        import caffe
        weights_dict = self.load_weights()
        net = caffe.Net(proto_path, caffe.TEST)
        print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
        num_conv = self.model.num_layers
        for l in range(num_conv):
            net = self.transfer_conv(l, net, weights_dict)
        if self.with_fc:
            for l in range(2):
                net = self.transfer_fc(l, net, weights_dict)
            net.params['fc8'][0].data[:] = weights_dict['fully_connected/weights'].transpose((1, 0))
            net.params['fc8'][1].data[:] = weights_dict['fully_connected/biases'].transpose()

        print('Saving Caffe model to: {}'.format(save_path))
        net.save(save_path)

    def transfer_conv(self, l, net, weights_dict):
        print('Layer {}'.format(l+1))
        if l == 0:
            weights = self.hwcn2nchw(weights_dict['conv_{}/weights'.format(l + 1)]) / self.scale
            if self.bgr:
                weights = weights[:, [2, 1, 0], :, :]   # Trained with RGB but caffe expects BGR!
        else:
            weights = self.hwcn2nchw(weights_dict['conv_{}/weights'.format(l + 1)])
        print('Weights shape: {}'.format(weights.shape))
        net.params['conv{}'.format(l + 1)][0].data[:] = weights
        net.params['conv{}'.format(l + 1)][1].data[:] = weights_dict['conv_{}/biases'.format(l + 1)]

        return net

    def transfer_fc(self, l, net, weights_dict):
        if l == 0:
            weights = weights_dict['fc{}/weights'.format(l + 1)]
            weights = weights.reshape(6, 6, 256, 4096)
            weights = self.hwcn2nchw(weights)
            weights = weights.reshape(4096, 9216)
            net.params['fc{}'.format(l + 6)][0].data[:] = weights
        else:
            net.params['fc{}'.format(l + 6)][0].data[:] = weights_dict['fc{}/weights'.format(l + 1)].transpose((1, 0))
        net.params['fc{}'.format(l + 6)][1].data[:] = weights_dict['fc{}/biases'.format(l + 1)]
        return net
