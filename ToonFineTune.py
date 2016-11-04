import os

from keras.preprocessing.image import ImageDataGenerator

from ToonNet import Classifier, make_name

# Training parameters
batch_size = 200
nb_epoch = 10

# Get the data-set object
from keras.datasets import cifar10

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

datagen = ImageDataGenerator()
num_layers = 3
num_res = 4
r_weight = None
net_load_name = make_name('ToonDiscriminator', num_res=num_res, num_layers=num_layers, r_weight=r_weight)

# Load the net
classifier = Classifier(input_shape=(32, 32, 3), num_layers=3, num_res=num_res, num_classes=10,
                        net_load_name=net_load_name)
classifier.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(classifier.name, 'CIFAR10')
print('Training network: {}'.format(net_name))

# Training
history = classifier.fit_generator(datagen.flow(X=X_train, y=y_train, batch_size=batch_size),
                                   samples_per_epoch=len(X_train),
                                   nb_epoch=nb_epoch,
                                   validation_data=datagen.flow(X=X_val, y=y_val, batch_size=batch_size),
                                   nb_val_samples=len(X_val),
                                   nb_worker=2,
                                   pickle_safe=True,
                                   max_q_size=16)

