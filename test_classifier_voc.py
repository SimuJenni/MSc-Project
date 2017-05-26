from Preprocessor import VOCPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.VOC2007 import VOC2007
from models.ToonNet_fcnobn import ToonNet

model = ToonNet(num_layers=5, batch_size=1)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=[227, 227, 3], augment_color=False, area_range=(0.1, 1.0))
tester = ToonNetTester(model, data, preprocessor, tag='supervised_bn_adam_restrart')
tester.test_classifier_voc(num_conv_trained=5)