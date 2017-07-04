from Preprocessor_new import Preprocessor
from eval.ToonNetTester_nofc9 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc9 import ToonNet

im_shape = [112, 112, 3]
model = ToonNet(num_layers=4, batch_size=128, im_shape=im_shape, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=im_shape, augment_color=True, hsv_color=False)
tester = ToonNetTester(model=model, dataset=data, pre_processor=preprocessor, tag='nofc9')
tester.test_reconstruction()
