from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_128 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_128 import ToonNet

model = ToonNet(num_layers=5, batch_size=64)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[128, 128, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='128',
                         lr_policy='const', optimizer='adam')
trainer.train()

