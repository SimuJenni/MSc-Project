import tensorflow as tf
import os

from ToonNet import VAEGAN, discriminator
from constants import LOG_DIR
from datasets import stl10

from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


data = stl10
model = VAEGAN(num_layers=4, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
LAYER_IDX = 0
FILTER_IDX = 0
LR = 0.001
NUM_STEPS = 20

x = tf.Variable(tf.random_uniform([1, 64, 64, 3], minval=-1.0, maxval=1.0), name='x')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    _, _, layers = discriminator(x, training=False, train_fc=False)
    ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
    vars = slim.get_variables_to_restore(include=['discriminator'])
    saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, ckpt.model_checkpoint_path)

    loss = tf.reduce_mean(layers[0][:, :, :, FILTER_IDX])
    var_grad = tf.gradients(loss, [x])[0]
    var_grad /= (tf.sqrt(tf.reduce_mean(tf.square(var_grad))) + 1e-5)

    for i in range(NUM_STEPS):
        var_grad_val = sess.run(var_grad)
        x += var_grad_val * LR

    img = x.eval()
    img = deprocess_image(img)
    imsave('%s_filter_%d.png' % (LAYER_IDX, FILTER_IDX), img)
