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
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.2

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

    # x += 1.0
    # x *= 255/2.
    # x = np.clip(x, 0, 255).astype('uint8')
    # return x


def max_activity_img(layer_id, filter_id, lr, ckpt, reuse=None):
    x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=10), name='x')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        _, _, layers = discriminator(x, training=False, train_fc=False, reuse=reuse)
        vars = slim.get_variables_to_restore(include=['discriminator'])
        saver = tf.train.Saver(var_list=vars)
        saver.restore(sess, ckpt.model_checkpoint_path)

        loss = -tf.reduce_sum(layers[layer_id - 1][:, :, :, filter_id])
        loss += 0.001*tf.reduce_sum(tf.square(x))
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        # grads_and_vars = opt.compute_gradients(loss, var_list=[x])
        # modded_grads_and_vars = [(-gv[0], gv[1]) for gv in grads_and_vars]
        # train_op = opt.apply_gradients(modded_grads_and_vars)
        train_op = opt.minimize(loss, var_list=[x])
        print('Layer: {} Filter: {} Learning-Rate: {}'.format(layer_id, filter_id, lr))

        for j in range(200):
            sess.run([train_op])
            with tf.control_dependencies([train_op]):
                # x *= (1. - 0.0001)
                x = clip_by_value(x, clip_value_min=-1., clip_value_max=1.)
                # print(sess.run(loss))

        img = x.eval()
        img = deprocess_image(np.squeeze(img))
        return img


data = imagenet
model = VAEGAN(num_layers=5, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
LAYER = 4
LR = 40
FILTERS = [i for i in range(9)]
imgs = [None for i in FILTERS]
for i, f in enumerate(FILTERS):
    if i == 0:
        reuse = None
    else:
        reuse = True
    imgs[i] = max_activity_img(LAYER, f, LR, ckpt, reuse=reuse)

montage_img = montage(imgs)
scipy.misc.toimage(montage_img, cmin=0, cmax=255).save('max_act_%d_new.png' % (LAYER))