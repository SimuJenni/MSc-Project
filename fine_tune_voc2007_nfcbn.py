from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet_nfcbn import ToonNet
from utils import get_checkpoint_path


model = ToonNet(num_layers=5, batch_size=16, fix_bn=False, vanilla_alex=True)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[227, 227, 3], augment_color=False, area_range=(0.1, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=250, tag='nfcbn',
                         lr_policy='voc', optimizer='sgd+momentum', init_lr=0.001)
chpt_path = get_checkpoint_path(trainer.get_save_dir())

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
