import numpy as np
import tensorflow as tf
from models.ToonNet_noBN import ToonNet_noBN as TNnoBN

from AlexNetConverter import AlexNetConverter
from Preprocessor import ImageNetPreprocessor
from datasets.ImageNet import ImageNet
from models.ToonNet import ToonNet as TN
from train.ToonNetTrainer import ToonNetTrainer

slim = tf.contrib.slim

test_nobn = True
ckpt = '../test_converter/model.ckpt-800722'

np.random.seed(42)
rand_num = np.float32(np.random.randn(96, 96, 3))

if test_nobn:
    model_dir = '../test_converter'
    sess = tf.Session()
    model = TNnoBN(num_layers=5, batch_size=128)
    converter1 = AlexNetConverter(model_dir, None, None, remove_bn=True, scale=1)

    with sess:
        disc_in = tf.constant(rand_num, shape=[1, 96, 96, 3])
        result, _ = model.discriminator.discriminate(disc_in, with_fc=True, training=False)
        sess.run(tf.global_variables_initializer())

        converter1.load_and_set_tf(model, sess)
        print(result.eval())

        saver = tf.train.Saver()
        save_path = saver.save(sess,  '../test_converter/model_transfer.ckpt')

else:
    model = TN(num_layers=5, batch_size=128)
    data = ImageNet()
    preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
    trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                             lr_policy='const', optimizer='adam')
    model_dir = '../test_converter'
    converter = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True, scale=1)
    with converter.sess:
        converter.extract_and_store_remove_batchnorm()
        result, _ = model.discriminator.discriminate(tf.constant(rand_num, shape=[1, 96, 96, 3]),
                                                     with_fc=True, reuse=True, training=False)
        print(result.eval())
