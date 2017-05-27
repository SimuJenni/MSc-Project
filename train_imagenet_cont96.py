from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_cont96 import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_cont96 import ToonNet

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=120, tag='cont96',
                         lr_policy='const', optimizer='adam')
trainer.train()

