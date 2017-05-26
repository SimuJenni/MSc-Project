from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet_64 import ToonNet
from utils import get_checkpoint_path

model = ToonNet(num_layers=5, batch_size=32, fix_bn=True)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=True, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='64_unsupervised_wbn',
                         lr_policy='voc', optimizer='sgd+momentum')
# chpt_path = get_checkpoint_path(trainer.get_save_dir())
# chpt_path = '/Data/Logs/ToonNet/imagenet_ToonNet64_default_3rd/model.ckpt-675542'
chpt_path = '/Data/Logs/ToonNet/imagenet_ToonNet_default_refactored_finetune_conv_5/model.ckpt-450360'

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
