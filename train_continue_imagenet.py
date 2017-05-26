import os

from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_cont import ToonNetTrainer
from constants import LOG_DIR
from datasets.ImageNet import ImageNet
from models.ToonNet_nobn import ToonNet

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=10, tag='continue_nobn_sgd_reinit',
                         lr_policy='voc', optimizer='sgd+momentum', init_lr=0.001, reinit_fc=True)

all_ckpt = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored/model.ckpt-800722')
disc_ckpt = os.path.join(LOG_DIR, 'batchnorm_removed/alexnet_nobn.ckpt')

trainer.train(all_ckpt, disc_ckpt)
