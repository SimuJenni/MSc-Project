from ToonNet_noBN import ToonNet_noBN
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import Preprocessor, distort_image
from utils import get_checkpoint_path
from datasets.VOC2007 import VOC2007
import tensorflow as tf


class VOCPreprocessor_nobn(Preprocessor):
    def __init__(self, target_shape, augment_color=True, aspect_ratio_range=(0.9, 1.1), area_range=(0.1, 1.0)):
        Preprocessor.__init__(self, target_shape, augment_color, aspect_ratio_range, area_range)

    def process_transfer_test(self, image):
        # Select random crops
        image = distort_image(image, self.target_shape[0], self.target_shape[1],
                              self.aspect_ratio_range, self.area_range)

        # Scale to [-1, 1]
        image = tf.to_float(image) - 127.5

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image

    def process_transfer_test(self, image):
        # Select random crops
        image = distort_image(image, self.target_shape[0], self.target_shape[1],
                              self.aspect_ratio_range, self.area_range)

        # Scale to [-1, 1]
        image = tf.to_float(image) - 127.5

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image


model = ToonNet_noBN(num_layers=5, batch_size=32)
data = ImageNet()
preprocessor = VOCPreprocessor_nobn(target_shape=[224, 224, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='continuation',
                         lr_policy='linear', optimizer='adam')
print(trainer.get_save_dir())
chpt_path = '/data/cvg/simon/data/Logs/logs_ToonNet/imagenet_ToonNet_default_continuation/model.ckpt-900811'

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
