import os

from ToonDataGenerator import ImageDataGenerator, Y2X_Y2Y
from ToonNet import Discriminator
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage


# Training parameters
batch_size = 200
nb_epoch = 10
num_layers = 3
num_res = 0

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator()

# Load the net
discriminator = Discriminator(input_shape=data.dims, num_layers=num_layers, num_res=num_res)
discriminator.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(discriminator.name, data.name)
print('Training network: {}'.format(net_name))

# Training
history = discriminator.fit_generator(
    datagen.flow_from_directory(data.train_dir, batch_size=batch_size, target_size=data.target_size, xy_fun=Y2X_Y2Y),
    samples_per_epoch=data.num_train,
    nb_epoch=nb_epoch,
    validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size, target_size=data.target_size,
                                                xy_fun=Y2X_Y2Y),
    nb_val_samples=data.num_val,
    nb_worker=4,
    pickle_safe=True,
    max_q_size=16)

# Save the model
discriminator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))

# Generate montage of sample-images
sample_size = 100
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=batch_size,
                                             target_size=data.target_size).next()
decoded_imgs = discriminator.predict(Y_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(net_name)))
