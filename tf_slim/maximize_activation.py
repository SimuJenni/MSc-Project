import tensorflow as tf
import os

from ToonNet import VAEGAN, discriminator
from constants import LOG_DIR
from datasets import imagenet
from tensorflow.python.ops.clip_ops import clip_by_value

from scipy.misc import imsave
from scipy.ndimage.filters import median_filter
import numpy as np
slim = tf.contrib.slim


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # x -= x.mean()
    # x /= (x.std() + 1e-5)
    # x *= 0.1
    #
    # # clip to [0, 1]
    # x += 0.5
    # x = np.clip(x, 0, 1)
    #
    # # convert to RGB array
    # x *= 255
    # x = np.clip(x, 0, 255).astype('uint8')
    # return x

    x += 1.0
    x *= 255/2.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


data = imagenet
model = VAEGAN(num_layers=5, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
LAYER_IDX = 3
FILTER_IDX = 11
LR = 10
NUM_STEPS = 300
l2decay = 0.001
medfilt_steps = 6

x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=10), name='x')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    _, _, layers = discriminator(x, training=False, train_fc=False)
    ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
    vars = slim.get_variables_to_restore(include=['discriminator'])
    saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, ckpt.model_checkpoint_path)

    loss = tf.reduce_sum(layers[LAYER_IDX-1][:, :, :, FILTER_IDX])
    opt = tf.train.GradientDescentOptimizer(learning_rate=LR)
    grads_and_vars = opt.compute_gradients(loss, var_list=[x])
    modded_grads_and_vars = [(-gv[0], gv[1]) for gv in grads_and_vars]
    train_op = opt.apply_gradients(modded_grads_and_vars)

    for i in range(NUM_STEPS):
        sess.run([train_op])
        with tf.control_dependencies([train_op]):
            x *= (1. - l2decay)
            x = clip_by_value(x, clip_value_min=-1., clip_value_max=1.)
            if not i % medfilt_steps:
                tmp = x.eval()
                assign_op = x.assign(median_filter(tmp, size=[1, 3, 3, 3]))
                sess.run(assign_op)
                print(sess.run(loss))

    img = x.eval()
    img = deprocess_image(np.squeeze(img))

    imsave('%s_filter_%d.png' % (LAYER_IDX, FILTER_IDX), img)
