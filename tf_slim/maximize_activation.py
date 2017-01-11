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
NUM_STEPS = 200

x = tf.Variable(tf.random_uniform([1, 64, 64, 3], minval=-1.0, maxval=1.0), name='x')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    _, _, layers = discriminator(x, training=False, train_fc=False)
    ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
    vars = slim.get_variables_to_restore(include=['discriminator'])
    saver = tf.train.Saver(var_list=vars)
    saver.restore(sess, ckpt.model_checkpoint_path)

    loss = -tf.reduce_mean(layers[LAYER_IDX][:, :, :, FILTER_IDX])
    optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss, var_list=[x])

    var_grad = tf.gradients(loss, [x])[0]

    for i in range(NUM_STEPS):
        _, var_grad_val = sess.run([optimizer, var_grad])
        x = clip_by_value(x, clip_value_min=-1., clip_value_max=1.)
        print(sess.run(loss))

    img = x.eval()
    print(img.shape)
    img = deprocess_image(np.squeeze(img))

    imsave('%s_filter_%d.png' % (LAYER_IDX, FILTER_IDX), img)
