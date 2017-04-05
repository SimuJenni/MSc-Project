from ToonNet import ToonNet
from datasets.STL10 import STL10
from ToonNet_Trainer import ToonNet_Trainer
from Preprocessor import Preprocessor


model = ToonNet(num_layers=4, batch_size=200, vgg_discriminator=True)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3])
trainer = ToonNet_Trainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='refactored',
                          lr_policy='const', optimizer='adam')
trainer.train()
