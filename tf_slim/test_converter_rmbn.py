from ToonNet_scaled import ToonNet
from datasets.ImageNet import ImageNet
from Preprocessor import ImageNetPreprocessor
from ToonNetTrainer import ToonNetTrainer
from AlexNetConverter import AlexNetConverter
import os
import numpy as np
import tensorflow as tf
import caffe
import skimage


def preprocess(img):
    out = np.copy(img)
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out = out.transpose((2, 0, 1))  # h, w, c -> c, h, w
    return out


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img -= 127
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (227, 227))
    return resized_img


model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                         lr_policy='const', optimizer='adam')

model_dir = '../../test_converter'
proto_path = '../deploy_scaled.prototxt'
ckpt = '../../test_converter/model.ckpt-800722'
save_path = os.path.join(model_dir, 'alexnet_v2_scaled.caffemodel')

np.random.seed(42)
img = load_image('../cat.jpg')

converter = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True, scale=1., bgr=True)
with converter.sess:
    converter.extract_and_store()
    result, _ = model.discriminator.discriminate(tf.constant(img, shape=[1, 227, 227, 3], dtype=tf.float32),
                                                 with_fc=False, reuse=True, training=False)
    result_tf = result.eval()

converter.load_and_set_caffe_weights(proto_path=proto_path, save_path=save_path)

net_caffe = caffe.Net(proto_path, save_path, caffe.TEST)

net_caffe.blobs['data'].data[0] = preprocess(img)
assert net_caffe.blobs['data'].data[0].shape == (3, 227, 227)
# show_caffe_net_input()
net_caffe.forward()
result_caffe = net_caffe.blobs['Convolution5'].data[0]
result_caffe = result_caffe.transpose((1, 2, 0))  # h, w, c -> c, h, w

print(np.linalg.norm(result_tf - result_caffe))

