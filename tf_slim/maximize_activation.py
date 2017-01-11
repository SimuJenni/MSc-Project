import tensorflow as tf
import os

from ToonNet import VAEGAN, discriminator
from constants import LOG_DIR
from datasets import imagenet
from tensorflow.python.ops.clip_ops import clip_by_value
from utils import montage

import numpy as np
import scipy.misc
slim = tf.contrib.slim


def deprocess_image(x):
    x += 1.0
    x *= 255/2.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def max_activity_img(layer_id, filter_id, lr, ckpt):
    x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=10), name='x')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        _, _, layers = discriminator(x, training=False, train_fc=False, reuse=True)
        vars = slim.get_variables_to_restore(include=['discriminator'], exclude=['discriminator/fully_connected'])
        saver = tf.train.Saver(var_list=vars)
        saver.restore(sess, ckpt.model_checkpoint_path)

        loss = tf.reduce_sum(layers[layer_id - 1][:, :, :, filter_id])
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        grads_and_vars = opt.compute_gradients(loss, var_list=[x])
        modded_grads_and_vars = [(-gv[0], gv[1]) for gv in grads_and_vars]
        train_op = opt.apply_gradients(modded_grads_and_vars)

        for i in range(200):
            sess.run([train_op])
            with tf.control_dependencies([train_op]):
                x *= (1. - 0.0001)
                x = clip_by_value(x, clip_value_min=-1., clip_value_max=1.)

        img = x.eval()
        img = deprocess_image(np.squeeze(img))
        return img


data = imagenet
model = VAEGAN(num_layers=5, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
LAYER = 1
LR = 1
FILTERS = [i for i in range(16)]
imgs = [None for i in FILTERS]
for i, f in enumerate(FILTERS):
    imgs[i] = max_activity_img(LAYER, f, LR, ckpt)

montage_img = montage(imgs)
scipy.misc.toimage(montage_img, cmin=0, cmax=255).save('max_act_%d.png' % (LAYER))
