from ToonNet import ToonNet
from datasets.STL10 import STL10
from ToonNet_Tester import ToonNet_Tester
from Preprocessor import Preprocessor

for fold in range(10):
    model = ToonNet(num_layers=4, batch_size=200)
    data = STL10()
    preprocessor = Preprocessor(target_shape=[96, 96, 3])
    tester = ToonNet_Tester(model, data, preprocessor, tag='refactored')
    tester.test_classifier_cv(num_conv_trained=5, fold=fold)
