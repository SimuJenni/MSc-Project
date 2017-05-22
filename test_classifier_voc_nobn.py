from Preprocessor import VOCPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.VOC2007 import VOC2007
from models.ToonNet_noBN import ToonNet

model = ToonNet(num_layers=5, batch_size=1)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=[224, 224, 3], augment_color=False, area_range=(0.1, 1.0))
tester = ToonNetTester(model, data, preprocessor, tag='nobn')
tester.test_classifier_voc(num_conv_trained=5)
