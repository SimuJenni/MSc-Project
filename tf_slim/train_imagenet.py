from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNet_Trainer import ToonNet_Trainer
from Preprocessor import ImageNetPreprocessor


model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNet_Trainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored')
trainer.train()
