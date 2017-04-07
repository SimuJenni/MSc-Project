import pickle

import tensorflow as tf
import caffe

from Preprocessor import ImageNetPreprocessor
from ToonNet import ToonNet
from ToonNet_Trainer import ToonNet_Trainer
from datasets.ImageNet import ImageNet

slim = tf.contrib.slim

class AlexNetConverter:
    def __init__(self, model_dir, model, net_id='discriminator'):
        self.model = model
        self.net_id = net_id
        self.bn_eps = 0.001
        self.weights_file = '{}/weights_dict'.format(model_dir)
        self.ckpt = tf.train.get_checkpoint_state(model_dir)

    def init_model(self, sess):
        x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=2), name='x')
        sess.run(tf.initialize_all_variables())
        model.discriminator.discriminate(x, with_fc=True)
        vars = slim.get_variables_to_restore(include=[self.net_id], exclude=['{}/fully_connected'.format(self.net_id)])
        saver = tf.train.Saver(var_list=vars)
        saver.restore(sess, self.ckpt)

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
        moving_mean, moving_variance, beta = self.get_bn_params(layer)
        weights = self.get_conv_weights(layer)
        new_weights = weights / tf.sqrt(moving_variance + self.bn_eps)
        if layer == 1:
            new_biases = self.get_conv_biases(layer) - moving_mean / tf.sqrt(moving_variance + self.bn_eps)
        else:
            new_biases = -moving_mean / tf.sqrt(moving_variance + self.bn_eps)

        return new_weights, new_biases

    def nhwc2nchw(self, input):
        return tf.transpose(input, [0, 3, 1, 2])

    def nchw2nhwc(self, input):
        return tf.transpose(input, [0, 2, 3, 1])

    def extract_and_store(self):
        num_conv = self.model.num_layers
        weights_dict = {}
        for l in range(num_conv):
            weights, biases = self.get_conv_rm_bn(l+1)
            weights_dict['conv_{}/weights'.format(l+1)] = weights
            weights_dict['conv_{}/biases'.format(l+1)] = biases
        self.save_weights(weights_dict)

    def save_weights(self, weights_dict):
        with open(self.weights_file + '.pkl', 'wb') as f:
            pickle.dump(weights_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_weights(self):
        with open(self.weights_file + '.pkl', 'rb') as f:
            return pickle.load(f)


model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNet_Trainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                          lr_policy='const', optimizer='adam')
model_dir = trainer.get_save_dir()

with trainer.sess.as_default():
    with trainer.graph.as_default():
        converter = AlexNetConverter(model_dir, model)
        converter.init_model(trainer.sess)
        converter.extract_and_store()
