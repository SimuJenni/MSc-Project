from Preprocessor import Preprocessor
from eval.ToonNetTester import ToonNetTester
from datasets.ImageNet import ImageNet
from models.ToonNet_new import ToonNet

model = ToonNet(num_layers=5, batch_size=64, vanilla_alex=True)
data = ImageNet()
preprocessor = Preprocessor(target_shape=[128, 128, 3])
tester = ToonNetTester(model, data, preprocessor, tag='NEW_correct')
tester.test_reconstruction()
