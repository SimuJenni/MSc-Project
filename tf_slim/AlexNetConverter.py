import pickle
import os

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


class AlexNetConverter:
    def __init__(self, model_dir, model, sess, remove_bn=False, ckpt=None, net_id='discriminator'):
        self.model = model
        self.sess = sess
        self.net_id = net_id
        self.bn_eps = 0.001
        self.weights_file = '{}/weights_dict'.format(model_dir)
        self.remove_bn = remove_bn
        if ckpt:
            self.ckpt = ckpt
        else:
            self.ckpt = tf.train.get_checkpoint_state(model_dir)

    def init_model(self):
        x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=2, seed=42), name='x')
        self.model.discriminator.discriminate(x, with_fc=False)
        #self.sess.run(tf.global_variables_initializer())
        vars = slim.get_variables_to_restore(include=[self.net_id], exclude=['{}/fully_connected'.format(self.net_id)])
        print('Variables: {}'.format([v.op.name for v in vars]))
        saver = tf.train.Saver(var_list=vars)
        saver.restore(self.sess, self.ckpt)

    def get_bn_params(self, layer):
        with tf.variable_scope(self.net_id, reuse=True):
            beta = slim.variable('conv_{}/BatchNorm/beta'.format(layer))
            moving_mean = slim.variable('conv_{}/BatchNorm/moving_mean'.format(layer))
            moving_variance = slim.variable('conv_{}/BatchNorm/moving_variance'.format(layer))
            return moving_mean, moving_variance, beta

    def get_conv_weights(self, layer):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable('conv_{}/weights'.format(layer))

    def get_conv_biases(self, layer):
        with tf.variable_scope(self.net_id, reuse=True):
            return slim.variable('conv_{}/biases'.format(layer))

    def get_conv_rm_bn(self, layer):
        weights = self.get_conv_weights(layer)
        if layer == 1:
            biases = self.get_conv_biases(layer)
            weights /= 255./2
        else:
            moving_mean, moving_variance, beta = self.get_bn_params(layer)
            alpha = tf.rsqrt(moving_variance + self.bn_eps)
            weights *= alpha
            biases = beta - moving_mean * alpha

        return weights, biases

    def get_conv_bn(self, layer):
        weights = self.get_conv_weights(layer)
        moving_mean = None
        moving_variance = None
        if layer == 1:
            biases = self.get_conv_biases(layer)
        else:
            moving_mean, moving_variance, beta = self.get_bn_params(layer)
            biases = beta

        return weights, biases, moving_mean, moving_variance

    def hwcn2nchw(self, input):
        return np.transpose(input, [3, 2, 0, 1])

    def nchw2hwcn(self, input):
        return np.transpose(input, [2, 3, 1, 0])

    def extract_and_store(self):
        with self.sess:
            self.init_model()
            num_conv = self.model.num_layers
            weights_dict = {}
            for l in range(num_conv):
                if self.remove_bn or l == 0:
                    weights, biases = self.get_conv_rm_bn(l+1)
                else:
                    weights, biases, moving_mean, moving_variance = self.get_conv_bn(l+1)
                    weights_dict['conv_{}_BN/mean'.format(l + 1)] = moving_mean.eval()
                    weights_dict['conv_{}_BN/variance'.format(l + 1)] = moving_variance.eval()
                weights_dict['conv_{}/weights'.format(l+1)] = weights.eval()
                weights_dict['conv_{}/biases'.format(l+1)] = biases.eval()
            self.save_weights(weights_dict)

    def save_weights(self, weights_dict):
        with open(self.weights_file + '.pkl', 'wb') as f:
            pickle.dump(weights_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self):
        with open(self.weights_file + '.pkl', 'rb') as f:
            return pickle.load(f)

    def load_and_set_caffe_weights(self, proto_path, save_dir):
        import caffe
        weights_dict = self.load_weights()
        net = caffe.Net(proto_path, caffe.TRAIN)
        print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
        num_conv = self.model.num_layers
        for l in range(num_conv):
            net = self.transfer(l, net, weights_dict)

        save_path = os.path.join(save_dir, 'alexnet_v2.caffemodel')
        print('Saving Caffe model to: {}'.format(save_path))
        net.save(save_path)

    def transfer_tf(self, other_model, sess):
        weights_dict = self.load_weights()
        with tf.variable_scope(self.net_id, reuse=True):
            for l in range(other_model.num_layers):
                weight_assign = slim.variable('conv_{}/weights'.format(l+1)).assign(
                    weights_dict['conv_{}/weights'.format(l+1)])
                bias_assign = slim.variable('conv_{}/biases'.format(l+1)).assign(
                    weights_dict['conv_{}/biases'.format(l+1)])
                sess.run(weight_assign)
                sess.run(bias_assign)

    def transfer(self, l, net, weights_dict):
        net.params['conv{}'.format(l + 1)][0].data[:] = self.hwcn2nchw(weights_dict['conv_{}/weights'.format(l + 1)])
        net.params['conv{}'.format(l + 1)][1].data[:] = weights_dict['conv_{}/biases'.format(l + 1)]
        if not self.remove_bn and l > 0:
            net.params['BatchNorm{}'.format(l)][0].data[:] = weights_dict['conv_{}_BN/mean'.format(l + 1)]
            net.params['BatchNorm{}'.format(l)][1].data[:] = weights_dict['conv_{}_BN/variance'.format(l + 1)]
        return net
