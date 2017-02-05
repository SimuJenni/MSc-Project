import tensorflow as tf
from preprocess import preprocess_imagenet
from scipy import misc
import numpy as np
from utils import montage

image = misc.imread('../catdog.jpg')
imgs_tr = [None for i in range(16)]

im = tf.placeholder(tf.float32, shape=np.shape(image))

im_p = preprocess_imagenet(image, output_height=160, output_width=160, augment_color=True)

with tf.Session() as sess:
    for i in range(16):
        imgs_tr[i] = sess.run(im_p, feed_dict={im: image})

mont = montage(imgs_tr)
misc.toimage(mont).save('test.png')
