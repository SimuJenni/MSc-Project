from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc13 import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet_nofc13 import ToonNet

model = ToonNet(num_layers=3, batch_size=200)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3], augment_color=True)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=300, tag='refactored',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=1e-8)
trainer.train_autoencoder()
trainer.train()