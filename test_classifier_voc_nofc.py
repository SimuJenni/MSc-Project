from Preprocessor_new import VOCPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.VOC2007 import VOC2007
from models.ToonNet_nofc12 import ToonNet

im_shape = [227, 227, 3]
model = ToonNet(num_layers=4, batch_size=1, fix_bn=False, im_shape=im_shape)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=im_shape, augment_color=False, area_range=(0.1, 1.0))
tester = ToonNetTester(model, data, preprocessor, tag='nofc12')
tester.test_classifier_voc(num_conv_trained=0)
