from Preprocessor import Preprocessor
from train.ToonNetTrainer import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet_stl import ToonNet
from utils import get_checkpoint_path

target_shape = [96, 96, 3]
model = ToonNet(num_layers=4, batch_size=200)
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='toon',
                       lr_policy='linear', optimizer='adam')
ckpt_path = get_checkpoint_path(trainer.get_save_dir())
trainer.transfer_finetune(ckpt_path, num_conv2train=5, num_conv2init=5)
