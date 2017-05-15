from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_4x4 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_4x4 import ToonNet

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=60, tag='4x4',
                         lr_policy='const', optimizer='adam')
trainer.train()

