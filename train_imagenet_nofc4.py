from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc4 import ToonNet

model = ToonNet(num_layers=5, batch_size=96, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[112, 112, 3], augment_color=True, hsv_color=False)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=135, tag='nofc4',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.0)
trainer.train()

