from Preprocessor import Preprocessor
from eval.ToonNetTester_128 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_128 import ToonNet

model = ToonNet(num_layers=5, batch_size=256)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[128, 128, 3])
tester = ToonNetTester(model, data, preprocessor, tag='128')
tester.test_reconstruction()
