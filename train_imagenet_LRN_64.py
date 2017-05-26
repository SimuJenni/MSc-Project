from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_LRN_64 import ToonNet

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[64, 64, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=135, tag='1st',
                         lr_policy='linear', optimizer='adam')
trainer.train()

