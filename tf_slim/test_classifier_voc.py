from ToonNet import ToonNet
from datasets.VOC2007 import VOC2007
from ToonNet_Tester import ToonNet_Tester
from Preprocessor import VOCPreprocessor

model = ToonNet(num_layers=5, batch_size=1)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3])
tester = ToonNet_Tester(model, data, preprocessor, tag='refactored')
tester.test_classifier_voc(num_conv_trained=5)
