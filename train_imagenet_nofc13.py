from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc13 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc13 import ToonNet

model = ToonNet(num_layers=4, batch_size=128, im_shape=[96, 96])
data = ImageNet()
preprocessor = Preprocessor(target_shape=[227, 227, 3], augment_color=True, hsv_color=False, area_range=(0.2, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=200, tag='nofc13',
                         lr_policy='linear', optimizer='adam', init_lr=0.0004, end_lr=1e-6)
trainer.train_autoencoder()

