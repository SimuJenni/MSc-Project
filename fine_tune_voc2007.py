from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet import ToonNet
from constants import LOG_DIR
import os

model = ToonNet(num_layers=5, batch_size=32, fix_bn=True)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=False, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='old2',
                         lr_policy='linear', optimizer='adam', init_lr=0.0002)
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored_finetune_conv_5/model.ckpt-450360')
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_cont96/model.ckpt-1210549')
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored/model.ckpt-800722')

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
