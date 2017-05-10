from ToonNet_noBN import ToonNet_noBN
from datasets.VOC2007 import VOC2007
from ToonNetTester import ToonNetTester
from Preprocessor import VOCPreprocessor

model = ToonNet_noBN(num_layers=5, batch_size=1)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3])
tester = ToonNetTester(model, data, preprocessor, tag='continuation')
tester.test_classifier_voc(num_conv_trained=5)
