from ToonNet_noBN import ToonNet_noBN
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import VOCPreprocessor
from datasets.VOC2007 import VOC2007

model = ToonNet_noBN(num_layers=5, batch_size=32)
data = ImageNet()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=False)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, tag='continuation',
                         lr_policy='linear', optimizer='adam')
print(trainer.get_save_dir())
chpt_path = '/Data/Logs/ToonNet/imagenet_ToonNet_default_continuation/model.ckpt-900811'

trainer.dataset = VOC2007()
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
