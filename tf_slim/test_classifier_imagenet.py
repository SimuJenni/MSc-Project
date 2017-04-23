from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from ToonNetTester import ToonNetTester
from Preprocessor import ImageNetPreprocessor

model = ToonNet(num_layers=5, batch_size=512)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[224, 224, 3])
tester = ToonNetTester(model, data, preprocessor, tag='refactored')
tester.test_classifier(num_conv_trained=5)
