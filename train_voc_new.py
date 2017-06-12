from Preprocessor import ImageNetPreprocessor
from train.ToonNetTrainer_new import ToonNetTrainer
from datasets.VOC2007 import VOC2007
from models.ToonNet_new import ToonNet
from constants import LOG_DIR
import os

model = ToonNet(num_layers=5, batch_size=32, vanilla_alex=True)
data = VOC2007()
preprocessor = ImageNetPreprocessor(target_shape=[128, 128, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=200, tag='NEW_correct4',
                         lr_policy='linear', optimizer='adam', init_lr=0.0002, end_lr=0.0001)
chpt_path = os.path.join(LOG_DIR, 'imagenet_ToonNet_default_NEW_correct3/model.ckpt-1236614')
trainer.train(chpt_path)

