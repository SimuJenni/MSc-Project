import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os

from ToonNet import Classifier, make_name
from datasets import CIFAR10, TinyImagenet

# Training parameters
batch_size = 100
nb_epoch = 20
num_layers = 3
num_res = 0
r_weight = 5.0
d_weight = 1.0
use_gan = True
use_gen = True
if use_gen:
    net_load_name = make_name('ToonGen', num_layers=num_layers)
else:
    net_load_name = make_name('ToonDisc', num_layers=num_layers)

if use_gan:
    net_load_name = '{}_gan'.format(net_load_name)

# Get the data-set object
data = CIFAR10()
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.75, 1.0],
    fill_mode='nearest',
    horizontal_flip=True
)

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
