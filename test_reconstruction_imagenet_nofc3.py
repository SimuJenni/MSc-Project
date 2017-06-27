from Preprocessor_new import Preprocessor
from eval.ToonNetTester_nofc import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_nofc3 import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[112, 112, 3], augment_color=True, hsv_color=False)
tester = ToonNetTester(model, data, preprocessor, tag='nofc3')
tester.test_reconstruction()
