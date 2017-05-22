from Preprocessor import VOCPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from datasets.VOC2007 import VOC2007
from models.ToonNet_noBN import ToonNet

model = ToonNet(num_layers=5, batch_size=32)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[227, 227, 3], augment_color=True, area_range=(0.2, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='nobn',
                         lr_policy='linear', optimizer='sgd+momentum', init_lr=0.001)
chpt_path = '/Data/Logs/ToonNet/test_convert/alexnet_nobn.ckpt'

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
