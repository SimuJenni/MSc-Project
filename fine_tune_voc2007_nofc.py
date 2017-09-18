from Preprocessor_new import VOCPreprocessor
from train.ToonNetTrainer_nofc12 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet_nofc12 import ToonNet
from utils import get_checkpoint_path

im_shape = [227, 227, 3]
model = ToonNet(num_layers=4, batch_size=16, fix_bn=False, im_shape=im_shape) 
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=im_shape, augment_color=True, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=250, tag='nofc12',
                         lr_policy='voc', optimizer='sgd', init_lr=0.001)
chpt_path = get_checkpoint_path(trainer.get_save_dir())

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=0, num_conv2init=5)
