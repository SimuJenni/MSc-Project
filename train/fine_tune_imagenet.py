from Preprocessor import ImageNetPreprocessor
from ToonNetTrainer import ToonNetTrainer
from datasets.ImageNet import ImageNet
from models.ToonNet import ToonNet
from utils import get_checkpoint_path

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='refactored',
                         lr_policy='linear', optimizer='sgd+momentum', init_lr=0.02)
chpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(chpt_path, num_conv2train=5, num_conv2init=0)
