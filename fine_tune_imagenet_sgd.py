from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_imnettest import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet_nobn import ToonNet
from constants import LOG_DIR
import os

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=5, tag='retrain_nobn_sgd',
                         lr_policy='voc', optimizer='sgd+momentum', init_lr=0.0002)
chpt_path = os.path.join(LOG_DIR, 'batchnorm_removed/alexnet_nobn_sup.ckpt')
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=5)
