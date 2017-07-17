from Preprocessor_new import VOCPreprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.VOC2007 import VOC2007
from models.ToonNet_nofc10 import ToonNet

im_shape = [224, 224, 3]
model = ToonNet(num_layers=5, batch_size=1, fix_bn=False, vanilla_alex=True, im_shape=im_shape)
data = VOC2007()
preprocessor = VOCPreprocessor(target_shape=im_shape, augment_color=False, area_range=(0.1, 1.0), hsv_color=False)
tester = ToonNetTester(model, data, preprocessor, tag='nofc10')
tester.test_classifier_voc(num_conv_trained=5)
