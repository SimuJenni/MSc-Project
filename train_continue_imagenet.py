import os

from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_cont import ToonNetTrainer
from constants import LOG_DIR
from datasets.ImageNet import ImageNet
from models.ToonNet import ToonNet
from constants import IMAGENET_EASY_TF_DATADIR

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet(cartoon_data_dir=IMAGENET_EASY_TF_DATADIR)
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=40,
                         tag='cont_essy', lr_policy='const', optimizer='adam', init_lr=0.0002,
                         reinit_fc=False)

all_ckpt = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_refactored/model.ckpt-800722')
# disc_ckpt = os.path.join(LOG_DIR, 'batchnorm_removed/alexnet_wobn.ckpt')

trainer.train_cont(all_ckpt, all_ckpt)
