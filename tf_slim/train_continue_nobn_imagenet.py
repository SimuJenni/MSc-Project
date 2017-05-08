from ToonNet_noBN import ToonNet_noBN
from datasets.ImageNet import ImageNet
from ToonNetTrainer_cont_nobn import ToonNetTrainer
from Preprocessor import ImageNetPreprocessor
import os
from constants import LOG_DIR


model = ToonNet_noBN(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=85, tag='continuation',
                         lr_policy='linear', optimizer='adam')
chpt_dir = os.path.join(LOG_DIR, 'test_convert/')
all_ckpt = os.path.join(chpt_dir, 'model.ckpt-800722')
disc_ckpt = os.path.join(chpt_dir, 'model_transfer.ckpt')
trainer.train(all_ckpt, disc_ckpt)
