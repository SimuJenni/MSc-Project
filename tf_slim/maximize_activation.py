import tensorflow as tf
import os

from ToonNet import VAEGAN, discriminator
from constants import LOG_DIR
from datasets import stl10
from tensorflow.python.ops.clip_ops import clip_by_value

from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # convert to RGB array
    x += 1.
    x *= 255./2
    x = np.clip(x, 0, 255).astype('uint8')
    return x


data = stl10
model = VAEGAN(num_layers=4, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
LAYER_IDX = 3
FILTER_IDX = 1
LR = 10
NUM_STEPS = 1000

x = tf.Variable(tf.random_uniform([1, 64, 64, 3], minval=-1.0, maxval=1.0), name='x')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    _, _, layers = discriminator(x, training=False, train_fc=False)
    ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
    vars = slim.get_variables_to_restore(include=['discriminator'])
    saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, ckpt.model_checkpoint_path)

    loss = tf.reduce_mean(layers[LAYER_IDX][:, :, :, FILTER_IDX])
    opt = tf.train.GradientDescentOptimizer(learning_rate=LR)
    grads_and_vars = opt.compute_gradients(loss, var_list=[x])
    modded_grads_and_vars = [(-gv[0], gv[1]) for gv in grads_and_vars]
    train_op = opt.apply_gradients(modded_grads_and_vars)

    for i in range(NUM_STEPS):
        sess.run([train_op])
        if not i % 100:
            with tf.control_dependencies([train_op]):
                print(sess.run(loss))

    img = x.eval()
    print(img.shape)
    img = deprocess_image(np.squeeze(img))

    imsave('%s_filter_%d.png' % (LAYER_IDX, FILTER_IDX), img)
