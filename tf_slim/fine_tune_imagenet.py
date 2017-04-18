from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNetTrainer import ToonNetTrainer
from Preprocessor import ImageNetPreprocessor
from utils import get_checkpoint_path

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=90, tag='refactored',
                         lr_policy='step', optimizer='sgd+momentum')
chpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(chpt_path, num_conv2train=0, num_conv2init=5)
