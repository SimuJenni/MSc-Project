from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_new import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_new import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=100, tag='NEW_correct_2',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003, end_lr=0.0001)
trainer.train()

