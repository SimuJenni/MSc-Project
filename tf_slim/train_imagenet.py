from ToonNet_scaled_lrn import ToonNet
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import ImageNetPreprocessor


model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                         lr_policy='const', optimizer='adam')
trainer.train()

