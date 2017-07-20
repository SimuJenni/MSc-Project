from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc13 import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet_nofc13 import ToonNet

model = ToonNet(num_layers=3, batch_size=200, im_shape=[64, 64, 3], pool5=False)
data = STL10()
preprocessor = Preprocessor(target_shape=[64, 64, 3], im_shape=[96, 96, 3], augment_color=True)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=300, tag='nofc13',
                         lr_policy='const', optimizer='adam', init_lr=0.0002)
# trainer.train_autoencoder()
trainer.train()