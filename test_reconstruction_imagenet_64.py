from Preprocessor import Preprocessor
from eval.ToonNetTester_64 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_64 import ToonNet

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[64, 64, 3])
tester = ToonNetTester(model, data, preprocessor, tag='3rd')
tester.test_reconstruction()