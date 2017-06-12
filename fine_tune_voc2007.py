from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet_new import ToonNet
from constants import LOG_DIR
import os

model = ToonNet(num_layers=5, batch_size=16, fix_bn=False, vanilla_alex=True)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[227, 227, 3], augment_color=False, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=250, tag='NEW_correct_15',
                         lr_policy='linear', optimizer='adam', init_lr=0.0002)
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored_finetune_conv_5/model.ckpt-450360')
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_cont96/model.ckpt-1210549')
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_vanilla_correct/model.ckpt-971559')
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_NEW_correct3/model.ckpt-1236614')


trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
