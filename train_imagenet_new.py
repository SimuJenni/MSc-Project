from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_new import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_new import ToonNet

model = ToonNet(num_layers=5, batch_size=64, vanilla_alex=True)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[128, 128, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=100, tag='NEW_correct',
                         lr_policy='linear', optimizer='adam', init_lr=0.0005)
trainer.train()

