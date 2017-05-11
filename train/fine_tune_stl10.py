from Preprocessor import Preprocessor
from ToonNetTrainer import ToonNetTrainer
from datasets.STL10 import STL10
from models.ToonNet import ToonNet
from utils import get_checkpoint_path

for fold in range(10):
    model = ToonNet(num_layers=4, batch_size=200, vgg_discriminator=True)
    data = STL10()
    preprocessor = Preprocessor(target_shape=[96, 96, 3])
    trainer = ToonNetTrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=400, tag='refactored',
                             lr_policy='linear', optimizer='adam')
    chpt_path = get_checkpoint_path(trainer.get_save_dir())
    trainer.finetune_cv(chpt_path, num_conv2train=5, num_conv2init=5, fold=fold)
