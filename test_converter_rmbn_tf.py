import os

import caffe
import numpy as np
import skimage
import tensorflow as tf
import tensorflow.contrib.slim as slim

from AlexNetConverter import AlexNetConverter
from Preprocessor import ImageNetPreprocessor
from datasets.ImageNet import ImageNet
from models.ToonNet import ToonNet
from models.ToonNet_nobn import ToonNet as TN_nb
from train.ToonNetTrainer import ToonNetTrainer
from utils import get_checkpoint_path

im_s = 96


def load_image(path):
    # load image
    img = np.float32(skimage.io.imread(path))
    img /= 127.5
    img -= 1.0
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (im_s, im_s))
    return resized_img


model = ToonNet(num_layers=5, batch_size=16)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[im_s, im_s, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                         lr_policy='const', optimizer='adam')

model_dir = '../test_converter'
ckpt = '../test_converter/model.ckpt-800722'

np.random.seed(42)
img = load_image('cat.jpg')

converter = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True, scale=1.0, bgr=True,
                             pad='SAME', im_size=(im_s, im_s), with_fc=False)
with converter.sess:
    converter.extract_and_store()
    net, encoded = model.discriminator.discriminate(tf.constant(img, shape=[1, im_s, im_s, 3], dtype=tf.float32),
                                                    with_fc=converter.with_fc, reuse=True, training=False,
                                                    pad=converter.pad)
    result_tf_1 = encoded.eval()

tf.reset_default_graph()
sess = tf.Session()
model_nb = TN_nb(num_layers=5, batch_size=16)

with sess:
    net, encoded = model_nb.discriminator.discriminate(tf.constant(img, shape=[1, im_s, im_s, 3], dtype=tf.float32),
                                                    with_fc=converter.with_fc, training=False,
                                                    pad=converter.pad)
    tf.global_variables_initializer()
    converter.load_and_set_tf(model_nb, sess)
    result_tf_2 = encoded.eval()
    saver = tf.train.Saver()
    save_path = saver.save(sess, "../test_converter/alexnet_nobn.ckpt")
    print(save_path)

print(np.linalg.norm(result_tf_1 - result_tf_2))
