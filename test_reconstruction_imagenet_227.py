from Preprocessor import ImageNetPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.ImageNet_227 import ImageNet
from models.ToonNet_224 import ToonNet

model = ToonNet(num_layers=5, batch_size=24)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
tester = ToonNetTester(model, data, preprocessor, tag='new')
tester.test_reconstruction()
