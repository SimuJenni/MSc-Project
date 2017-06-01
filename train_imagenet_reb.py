from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_reb import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_reb import ToonNet
from constants import IMAGENET_EASY_TF_DATADIR

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet(cartoon_data_dir=IMAGENET_EASY_TF_DATADIR)
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='rebuttal',
                         lr_policy='linear', optimizer='adam', init_lr=0.0003)
trainer.train()
