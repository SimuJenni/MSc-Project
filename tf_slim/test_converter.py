from ToonNet import ToonNet
from datasets.ImageNet import ImageNet
from Preprocessor import ImageNetPreprocessor
from ToonNet_Trainer import ToonNet_Trainer
from AlexNetConverter import AlexNetConverter

model = ToonNet(num_layers=5, batch_size=128)
data = ImageNet()
preprocessor = ImageNetPreprocessor(target_shape=[96, 96, 3])
trainer = ToonNet_Trainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=80, tag='refactored',
                          lr_policy='const', optimizer='adam')

model_dir = trainer.get_save_dir()
model_dir = '../../test_converter'
ckpt = '../../test_converter/model.ckpt-800722'

converter = AlexNetConverter(model_dir, model, trainer.sess, ckpt=ckpt, remove_bn=True)
converter.extract_and_store()

weights_dict = converter.load_weights()
print(weights_dict.keys())

converter.load_and_set_caffe_weights(proto_path='../deploy.prototxt', save_dir=model_dir)
