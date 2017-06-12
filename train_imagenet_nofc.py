from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nofc import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='nofc',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.0001)
trainer.train()

