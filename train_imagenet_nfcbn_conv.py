from Preprocessor_new import Preprocessor
from train.ToonNetTrainer_nfcbn import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nfcbn import ToonNet

model = ToonNet(num_layers=5, batch_size=96, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[112, 112, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=100, tag='nfcbn_conv',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.0001)
trainer.train()

