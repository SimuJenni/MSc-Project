from Preprocessor import Preprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet_stl import ToonNet

model = ToonNet(num_layers=4, batch_size=200)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3], augment_color=True)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='toon',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003)
trainer.train()