from keras.preprocessing.image import ImageDataGenerator

from ToonNet import Classifier, make_name
from datasets import CIFAR10

# Training parameters
batch_size = 200
nb_epoch = 10
num_layers = 3
num_res = 4
r_weight = None
net_load_name = make_name('ToonDiscriminator', num_res=num_res, num_layers=num_layers, r_weight=r_weight)

# Get the data-set object
data = CIFAR10()
datagen = ImageDataGenerator()

# Load the net
classifier = Classifier(input_shape=data.dims, num_layers=3, num_res=num_res, num_classes=data.num_classes,
                        net_load_name=net_load_name)
classifier.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(classifier.name, 'CIFAR10')
print('Training network: {}'.format(net_name))

# Training
history = classifier.fit_generator(
    datagen.flow_from_directory(data.train_dir, target_size=data.target_size, class_mode='binary',
                                batch_size=batch_size),
    samples_per_epoch=data.num_train,
    nb_epoch=nb_epoch,
    validation_data=datagen.flow_from_directory(data.val_dir, target_size=data.target_size, class_mode='binary',
                                                batch_size=batch_size),
    nb_val_samples=data.num_val,
    nb_worker=2,
    pickle_safe=False,
    max_q_size=8)
