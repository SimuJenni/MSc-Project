from Preprocessor import VOCPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.VOC2007 import VOC2007
from models.ToonNet_new import ToonNet

model = ToonNet(num_layers=5, batch_size=1, fix_bn=True, vanilla_alex=True)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=[227, 227, 3], augment_color=False, area_range=(0.1, 1.0))
tester = ToonNetTester(model, data, preprocessor, tag='NEW_correct_17')
tester.test_classifier_voc(num_conv_trained=5)
