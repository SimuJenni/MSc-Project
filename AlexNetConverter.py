import pickle
import os

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


class AlexNetConverter:
    def __init__(self, model_dir, model, sess, remove_bn=False, ckpt=None, net_id='discriminator', scale=1.0, bgr=False,
                 exclude=None, with_fc=False, pad='VALID', im_size=(224, 224)):
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
        if ckpt:
            self.ckpt = ckpt
        else:
            self.ckpt = tf.train.get_checkpoint_state(model_dir)

    def init_model(self):
        x = tf.Variable(tf.random_normal([1, self.im_size[0], self.im_size[1], 3], stddev=2, seed=42), name='x')
        self.model.discriminator.discriminate(x, with_fc=self.with_fc, training=False, pad=self.pad)
        self.sess.run(tf.global_variables_initializer())
        var2restore = slim.get_variables_to_restore(include=[self.net_id], exclude=self.exclude)
        print('Variables: {}'.format([v.op.name for v in var2restore]))
        saver = tf.train.Saver(var_list=var2restore)
        saver.restore(self.sess, self.ckpt)

    def get_bn_params(self, layer_id):
        print('Extracting {}/BatchNorm'.format(layer_id))
        with tf.variable_scope(self.net_id, reuse=True):
            beta = slim.variable('{}/BatchNorm/beta'.format(layer_id))
            moving_mean = slim.variable('{}/BatchNorm/moving_mean'.format(layer_id))
            moving_variance = slim.variable('{}/BatchNorm/moving_variance'.format(layer_id))
            return moving_mean, moving_variance, beta

    def get_conv_weights(self, layer):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable('conv_{}/weights'.format(layer))

    def get_conv_biases(self, layer):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable('conv_{}/biases'.format(layer))

    def get_fc_weights(self, layer=None):
        with tf.variable_scope(self.net_id, reuse=True):
            if layer:
                return slim.variable('fc{}/weights'.format(layer))
            else:
                return slim.variable('fully_connected/weights')

    def get_fc_biases(self, layer=None):
        with tf.variable_scope(self.net_id, reuse=True):
            if layer:
                return slim.variable('fc{}/biases'.format(layer))
            else:
                return slim.variable('fully_connected/biases')

    def get_conv(self, layer):
        weights = self.get_conv_weights(layer)
        biases = self.get_conv_biases(layer)
        return weights, biases

    def get_conv_rm_bn(self, layer):
        weights = self.get_conv_weights(layer)
        moving_mean, moving_variance, beta = self.get_bn_params('conv_{}'.format(layer))
        alpha = tf.rsqrt(moving_variance + self.bn_eps)
        print('Mean alpha at layer {}: {}'.format(layer, np.mean(alpha.eval())))
        weights *= alpha
        biases = beta - moving_mean * alpha
        return weights, biases

    def get_fc(self, layer=None):
        weights = self.get_fc_weights(layer)
        biases = self.get_fc_biases(layer)
        return weights, biases

    def get_fc_rm_bn(self, layer):
        weights = self.get_fc_weights(layer)
        moving_mean, moving_variance, beta = self.get_bn_params('fc{}'.format(layer))
        alpha = tf.rsqrt(moving_variance + self.bn_eps)
        weights *= alpha
        biases = beta - moving_mean * alpha
        return weights, biases

    def hwcn2nchw(self, input):
        return np.transpose(input, [3, 2, 0, 1])

    def nchw2hwcn(self, input):
        return np.transpose(input, [2, 3, 1, 0])

    def extract_and_store_remove_batchnorm(self):
        self.init_model()
        num_conv = self.model.num_layers
        weights_dict = {}
        for l in range(num_conv):
            if l == 0:
                weights, biases = self.get_conv(l+1)
            else:
                weights, biases = self.get_conv_rm_bn(l+1)
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

    def extract_and_store_keep_batchnorm(self):
        self.init_model()
        num_conv = self.model.num_layers
        weights_dict = {}
        for l in range(num_conv):
            if l == 0:
                weights, biases = self.get_conv(l+1)
            else:
                weights = self.get_conv_weights(l+1)
                moving_mean, moving_variance, beta = self.get_bn_params('conv_{}'.format(l+1))
                weights_dict['conv_{}_BN/mean'.format(l + 1)] = moving_mean.eval()
                weights_dict['conv_{}_BN/variance'.format(l + 1)] = moving_variance.eval()
                weights_dict['conv_{}_BN/beta'.format(l + 1)] = beta.eval()
                biases = tf.zeros_like(beta)
            weights_dict['conv_{}/weights'.format(l + 1)] = weights.eval()
            weights_dict['conv_{}/biases'.format(l + 1)] = biases.eval()
        if self.with_fc:

            for l in range(2):
                weights = self.get_fc_weights(l+1)
                moving_mean, moving_variance, beta = self.get_bn_params('fc{}'.format(l + 1))
                weights_dict['fc{}_BN/mean'.format(l + 1)] = moving_mean.eval()
                weights_dict['fc{}_BN/variance'.format(l + 1)] = moving_variance.eval()
                weights_dict['fc{}_BN/beta'.format(l + 1)] = beta.eval()
                biases = tf.zeros_like(beta)
                weights_dict['fc{}/weights'.format(l+1)] = weights.eval()
                weights_dict['fc{}/biases'.format(l+1)] = biases.eval()
            weights, biases = self.get_fc()
            weights_dict['fully_connected/weights'] = weights.eval()
            weights_dict['fully_connected/biases'] = biases.eval()
        self.save_weights(weights_dict)

    def extract_and_store_nobn(self):
        self.init_model()
        num_conv = self.model.num_layers
        weights_dict = {}
        for l in range(num_conv):
            weights, biases = self.get_conv(l+1)
            weights_dict['conv_{}/weights'.format(l+1)] = weights.eval()
            weights_dict['conv_{}/biases'.format(l+1)] = biases.eval()
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

    def load_and_set_tf(self, other_model, sess):
        weights_dict = self.load_weights()
        with tf.variable_scope(self.net_id, reuse=True):
            for l in range(other_model.num_layers):
                weight_assign = slim.variable('conv_{}/weights'.format(l+1)).assign(
                    weights_dict['conv_{}/weights'.format(l+1)])
                bias_assign = slim.variable('conv_{}/biases'.format(l+1)).assign(
                    weights_dict['conv_{}/biases'.format(l+1)])
                sess.run(weight_assign)
                sess.run(bias_assign)
            for l in range(2):
                weight_assign = slim.variable('fc{}/weights'.format(l+1)).assign(
                    weights_dict['fc{}/weights'.format(l+1)])
                bias_assign = slim.variable('fc{}/biases'.format(l+1)).assign(
                    weights_dict['fc{}/biases'.format(l+1)])
                sess.run(weight_assign)
                sess.run(bias_assign)
            weight_assign = slim.variable('fully_connected/weights').assign(weights_dict['fully_connected/weights'])
            bias_assign = slim.variable('fully_connected/biases').assign(weights_dict['fully_connected/biases'])
            sess.run(weight_assign)
            sess.run(bias_assign)

    def transfer_conv(self, l, net, weights_dict):
        if l == 0:
            weights = self.hwcn2nchw(weights_dict['conv_{}/weights'.format(l + 1)]) / self.scale
            if self.bgr:
                weights = weights[:, [2, 1, 0], :, :]   # Trained with RGB but caffe expects BGR!
        else:
            weights = self.hwcn2nchw(weights_dict['conv_{}/weights'.format(l + 1)])
        net.params['conv{}'.format(l + 1)][0].data[:] = weights
        if not self.remove_bn and l > 0:
            net.params['BatchNorm{}'.format(l+1)][2].data[:] = 1
            net.params['BatchNorm{}'.format(l+1)][0].data[:] = weights_dict['conv_{}_BN/mean'.format(l + 1)]
            net.params['BatchNorm{}'.format(l+1)][1].data[:] = weights_dict['conv_{}_BN/variance'.format(l + 1)]
            net.params['Scale{}'.format(l+1)][1].data[:] = weights_dict['conv_{}_BN/beta'.format(l + 1)]
        else:
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
        if not self.remove_bn:
            net.params['BatchNorm{}'.format(l+6)][2].data[:] = 1
            net.params['BatchNorm{}'.format(l+6)][0].data[:] = weights_dict['fc{}_BN/mean'.format(l + 1)]
            net.params['BatchNorm{}'.format(l+6)][1].data[:] = weights_dict['fc{}_BN/variance'.format(l + 1)]
            net.params['Scale{}'.format(l+6)][1].data[:] = weights_dict['fc{}_BN/beta'.format(l + 1)]
        return net