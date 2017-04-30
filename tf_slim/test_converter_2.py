from ToonNet_noBN import ToonNet_noBN as TNnoBN
from ToonNet import ToonNet as TN
import numpy as np

import tensorflow as tf
from datasets.ImageNet import ImageNet
from Preprocessor import ImageNetPreprocessor
from ToonNetTrainer import ToonNetTrainer
from AlexNetConverter import AlexNetConverter


test_nobn = True

np.random.seed(42)
rand_num = np.float32(np.random.randn(128 * 128 * 3))
print(rand_num)

if test_nobn:
    model_dir = '../../test_converter'
    sess = tf.Session()
    model = TNnoBN(num_layers=5, batch_size=128)
    converter1 = AlexNetConverter(model_dir, None, None, remove_bn=True)

    with sess:
        result, _ = model.discriminator.discriminate(tf.constant(rand_num*255./2, shape=[1, 128, 128, 3]), with_fc=False)
        sess.run(tf.global_variables_initializer())
        converter1.transfer_tf(model, sess)
        print(result.eval()[0, 0, 0, :10])

else:
    model = TN(num_layers=5, batch_size=128)
    data = ImageNet()
    preprocessor = ImageNetPreprocessor(target_shape=[96, 128, 3])
    trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                             lr_policy='const', optimizer='adam')
    model_dir = '../../test_converter'
    ckpt = '../../test_converter/model.ckpt-800722'
    converter2 = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True)
    converter2.init_model()
    with converter2.sess:
        result, _ = model.discriminator.discriminate(tf.constant(rand_num, shape=[1, 128, 128, 3]),
                                                     with_fc=False, reuse=True, training=False)
        print(result.eval()[0, 0, 0, :10])
