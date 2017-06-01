from Preprocessor import Preprocessor
from eval.ToonNetTester_reb import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_reb import ToonNet
from constants import IMAGENET_EASY_TF_DATADIR

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet(cartoon_data_dir=IMAGENET_EASY_TF_DATADIR)
preprocessor = Preprocessor(target_shape=[96, 96, 3])
tester = ToonNetTester(model, data, preprocessor, tag='rebuttal_new')
tester.test_reconstruction()
