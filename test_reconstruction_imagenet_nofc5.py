from Preprocessor_new import Preprocessor
from eval.ToonNetTester_nofc5 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc5 import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[64, 64, 3], augment_color=False, hsv_color=False)
tester = ToonNetTester(model, data, preprocessor, tag='nofc5')
tester.test_reconstruction()
