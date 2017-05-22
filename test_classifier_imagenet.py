from Preprocessor import ImageNetPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet import ToonNet

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
tester = ToonNetTester(model, data, preprocessor, tag='refactored')
tester.test_classifier(num_conv_trained=5)
