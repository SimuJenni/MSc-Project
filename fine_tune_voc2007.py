from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet import ToonNet
from utils import get_checkpoint_path

model = ToonNet(num_layers=5, batch_size=32, fix_bn=False)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=True, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='unsupervised_wbn',
                         lr_policy='linear', optimizer='adam')
# chpt_path = get_checkpoint_path(trainer.get_save_dir())
chpt_path = '/Data/Logs/ToonNet/imagenet_ToonNet_default_refactored/model.ckpt-800722'
# chpt_path = '/Data/Logs/ToonNet/imagenet_ToonNet_default_refactored_finetune_conv_5/model.ckpt-450360'

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
