from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_64 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_64 import ToonNet

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[64, 64, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='second',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.0001)
trainer.train()

