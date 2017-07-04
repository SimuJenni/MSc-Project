from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc10 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc10 import ToonNet

im_shape = [224, 224, 3]
model = ToonNet(num_layers=5, batch_size=64, im_shape=im_shape, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=im_shape, augment_color=True, hsv_color=False)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=180, tag='nofc10',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=1e-6)
trainer.train()

