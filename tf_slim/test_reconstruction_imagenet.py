from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNet_Tester import ToonNet_Tester
from Preprocessor import Preprocessor

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
tester = ToonNet_Tester(model, data, preprocessor, tag='refactored')
tester.test_reconstruction()
