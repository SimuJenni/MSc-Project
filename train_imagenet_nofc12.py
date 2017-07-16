from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc10 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc12 import ToonNet

im_shape = [112, 112, 3]
model = ToonNet(num_layers=4, batch_size=128, im_shape=im_shape, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=im_shape, augment_color=True, hsv_color=False, area_range=(0.25, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=160, tag='nofc12',
                         lr_policy='linear', optimizer='adam', init_lr=0.0004, end_lr=1e-6)
trainer.train()

