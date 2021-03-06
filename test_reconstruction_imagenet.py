from Preprocessor import Preprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_vanilla import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
tester = ToonNetTester(model, data, preprocessor, tag='vanilla_correct')
tester.test_reconstruction()
