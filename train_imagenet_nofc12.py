from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc12 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc12 import ToonNet

model = ToonNet(num_layers=4, batch_size=128, im_shape=[112, 112])
data = ImageNet()
preprocessor = Preprocessor(target_shape=[227, 227, 3], im_shape=[256, 256, 3], augment_color=True,
                            area_range=(0.2, 1.0))
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=200, tag='nofc12',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=1e-6)
trainer.train()

