import os

from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_cont_nobn import ToonNetTrainer
import constants
from datasets.ImageNet import ImageNet
from models.ToonNet_noBN import ToonNet_noBN

model = ToonNet_noBN(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[227, 227, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=30, tag='227',
                         lr_policy='linear', optimizer='adam')
print(trainer.get_save_dir())
all_ckpt ='/Data/Logs/ToonNet/imagenet_ToonNet_default_continuation/model.ckpt-900811'
trainer.train(all_ckpt, all_ckpt)