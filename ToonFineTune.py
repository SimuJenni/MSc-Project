import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os

from ToonNet import Classifier, make_name
from datasets import CIFAR10, TinyImagenet

# Training parameters
batch_size = 100
nb_epoch = 10
num_layers = 3
num_res = 0
r_weight = 1.0
d_weight = 1.0
use_gan = True
use_gen = True
if use_gen:
    if use_gan:
        net_load_name = make_name('gGAN2', num_res=num_res, num_layers=[num_layers, num_layers], r_weight=r_weight, d_weight=d_weight)
    else:
        net_load_name = make_name('ToonGenerator', num_res=num_res, num_layers=num_layers)
else:
    if use_gan:
        net_load_name = make_name('dGAN2', num_res=num_res, num_layers=[num_layers, num_layers], r_weight=r_weight, d_weight=d_weight)
    else:
        net_load_name = make_name('ToonDiscriminator', num_res=num_res, num_layers=num_layers)
# net_load_name = None

# Get the data-set object
data = CIFAR10()
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=[0.9, 1.0],
                             horizontal_flip=True,
                             fill_mode='nearest')

# Load the net
classifier = Classifier(input_shape=data.dims, num_layers=num_layers, num_res=num_res, num_classes=data.num_classes,
                        net_load_name=net_load_name, use_gen=use_gen, batch_size=batch_size)
classifier.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(classifier.name, data.name)
print('Training network: {}'.format(net_name))

# Training
history = classifier.fit_generator(
    datagen.flow_from_directory(data.train_dir, target_size=data.target_size, batch_size=batch_size),
    samples_per_epoch=data.num_train,
    nb_epoch=nb_epoch,
    validation_data=datagen.flow_from_directory(data.val_dir, target_size=data.target_size, batch_size=batch_size),
    nb_val_samples=data.num_val,
    nb_worker=2,
    pickle_safe=True,
    max_q_size=16)


class_dirs = os.listdir(data.val_dir)
y_true = []
for i, c in enumerate(class_dirs):
    y_true += [i]*len(os.listdir(os.path.join(data.val_dir, c)))

pred = classifier.predict_generator(generator=datagen.flow_from_directory(data.val_dir, target_size=data.target_size,
                                                                          batch_size=batch_size, shuffle=False),
                                    val_samples=data.num_val)
y_pred = np.argmax(pred, axis=1)
print(np.mean(np.equal(y_true, y_pred)))
