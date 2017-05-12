import os

from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_cont_nobn_227 import ToonNetTrainer
from constants import LOG_DIR
from datasets.ImageNet_227 import ImageNet
from models.ToonNet_224 import ToonNet

model = ToonNet(num_layers=5, batch_size=24)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='new',
                         lr_policy='linear', optimizer='adam', init_lr=0.0002)

all_ckpt = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored/model.ckpt-800722')
trainer.train(all_ckpt, all_ckpt)