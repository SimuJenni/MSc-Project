from Preprocessor import Preprocessor
from ToonNetTrainer import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet import ToonNet

model = ToonNet(num_layers=4, batch_size=200, vgg_discriminator=True)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='refactored',
                         lr_policy='const', optimizer='adam')
trainer.train()
