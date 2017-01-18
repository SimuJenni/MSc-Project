import tensorflow as tf
from preprocess import preprocess_finetune_train
from scipy import misc
import numpy as np
from utils import montage


image = misc.imread('../catdog.jpg')
imgs_tr = [None for i in range(16)]

im = tf.placeholder(tf.float32, shape=np.shape(image))
im_p = preprocess_finetune_train(image, output_height=160, output_width=160, augment_color=True, resize_side_max=192, resize_side_min=192)

with tf.Session() as sess:

    for i in range(16):
        imgs_tr[i] = sess.run(im_p, feed_dict={im: image})

mont = montage(imgs_tr)
misc.toimage(mont).save('test.png')
