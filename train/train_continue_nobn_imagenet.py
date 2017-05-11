import os

from Preprocessor import ImageNetPreprocessor
from ToonNetTrainer_cont_nobn import ToonNetTrainer
from constants import LOG_DIR
from datasets.ImageNet import ImageNet
from models.ToonNet_noBN import ToonNet_noBN

model = ToonNet_noBN(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='continuation',
                         lr_policy='const', optimizer='adam', init_lr=0.0001)
chpt_dir = os.path.join(LOG_DIR, 'test_convert/')
all_ckpt = os.path.join(chpt_dir, 'model.ckpt-800722')
disc_ckpt = os.path.join(chpt_dir, 'model_transfer.ckpt')
trainer.train(all_ckpt, disc_ckpt)
