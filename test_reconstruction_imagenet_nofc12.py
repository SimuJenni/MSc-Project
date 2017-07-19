from Preprocessor_new import Preprocessor
from eval.ToonNetTester_nofc12 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc12 import ToonNet

model = ToonNet(num_layers=4, batch_size=128, im_shape=[96, 96])
data = ImageNet()
preprocessor = Preprocessor(target_shape=[227, 227, 3], im_shape=[256, 256, 3], augment_color=True)
tester = ToonNetTester(model=model, dataset=data, pre_processor=preprocessor, tag='nofc12')
tester.test_reconstruction()
