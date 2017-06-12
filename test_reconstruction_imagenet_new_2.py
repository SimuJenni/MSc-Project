from Preprocessor import Preprocessor
from eval.ToonNetTester_new_2 import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_new_2 import ToonNet

model = ToonNet(num_layers=5, batch_size=128, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[96, 96, 3])
tester = ToonNetTester(model, data, preprocessor, tag='NEW_correct_2')
tester.test_reconstruction()
