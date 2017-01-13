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
    x /= np.std(x)
    x *= 0.2
    x += 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def max_activity_img(layer_id, num_filters, lr, ckpt, reuse=None):
    imgs = [None for i in range(num_filters)]
    losses = [0. for i in range(num_filters)]
    x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=2), name='x')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.initialize_all_variables())
        sess.run(x.assign(tf.random_normal([1, 128, 128, 3], stddev=2)))

        _, _, layers = discriminator(x, training=False, train_fc=False, reuse=reuse)
        vars = slim.get_variables_to_restore(include=['discriminator'], exclude=['discriminator/fully_connected'])
        saver = tf.train.Saver(var_list=vars)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for f in range(num_filters):
            sess.run(x.assign(tf.random_normal([1, 128, 128, 3], stddev=2)))
            loss = -tf.reduce_sum(layers[layer_id - 1][:, :, :, f])
            loss += 0.0001*tf.reduce_sum(tf.square(x))
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
            train_op = opt.minimize(loss, var_list=[x])
            print('Layer: {} Filter: {} Learning-Rate: {}'.format(layer_id, f, lr))

            for j in range(200):
                tmp = x.eval()
                tmp = clip_by_value(tmp, clip_value_min=-1., clip_value_max=1.)
                sess.run(x.assign(tmp))
                sess.run([train_op])

            img = x.eval()
            imgs[f] = deprocess_image(np.squeeze(img))
            losses[f] = sess.run(loss)
        return imgs, losses

data = imagenet
model = VAEGAN(num_layers=5, batch_size=1, data_size=1, num_epochs=1)
MODLE_DIR = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
ckpt = tf.train.get_checkpoint_state(MODLE_DIR)
LAYER = 3
LR = 1

imgs, losses = max_activity_img(LAYER, 16, LR, ckpt)
print(losses)

imgs = [x for (y,x) in sorted(zip(losses, imgs))]
montage_img = montage(imgs)
scipy.misc.toimage(montage_img, cmin=0, cmax=255).save('max_act_%d_test_new1.png' % (LAYER))
