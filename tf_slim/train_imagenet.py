from ToonNet_noBN import ToonNet_noBN
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import ImageNetPreprocessorNoScale


model = ToonNet_noBN(num_layers=5, batch_size=192)
data = ImageNet()
preprocessor = ImageNetPreprocessorNoScale(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=100, tag='refactored',
                         lr_policy='const', optimizer='adam')
trainer.train()
