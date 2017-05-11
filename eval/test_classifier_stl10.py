from Preprocessor import Preprocessor
from ToonNetTester import ToonNetTester
from datasets.STL10 import STL10
from models.ToonNet import ToonNet

for fold in range(10):
    model = ToonNet(num_layers=4, batch_size=200, vgg_discriminator=True)
    data = STL10()
    preprocessor = Preprocessor(target_shape=[96, 96, 3])
    tester = ToonNetTester(model, data, preprocessor, tag='refactored')
    tester.test_classifier_cv(num_conv_trained=5, fold=fold)
