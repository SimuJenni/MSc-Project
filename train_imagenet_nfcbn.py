from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nfcbn import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nfcbn import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='nfcbn2',
                         lr_policy='linear', optimizer='RMSProp', init_lr=0.0005, end_lr=0.0001)
trainer.train()

