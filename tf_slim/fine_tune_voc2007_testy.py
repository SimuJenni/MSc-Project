from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import VOCPreprocessor
from utils import get_checkpoint_path
from datasets.VOC2007 import VOC2007

model = ToonNet(num_layers=5, batch_size=32, fix_bn=True)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=False)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='refactored',
                         lr_policy='step', optimizer='sgd+momentum')
chpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
