from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNetTester import ToonNetTester
from Preprocessor import Preprocessor

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
tester = ToonNetTester(model, data, preprocessor, tag='refactored')
tester.test_reconstruction()
