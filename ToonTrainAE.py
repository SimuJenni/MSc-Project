import os

from ToonDataGenerator import ImageDataGenerator
from ToonNet import Gen
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage


# Training parameters
batch_size = 200
nb_epoch = 5
num_layers = 4

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=[0.9, 1.0],
    fill_mode='nearest',
    horizontal_flip=True
)

# Load the net
generator = Gen(input_shape=data.dims, num_layers=num_layers, batch_size=batch_size)
generator.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(generator.name, data.name)
print('Training network: {}'.format(net_name))

# Training
history = generator.fit_generator(
    datagen.flow_from_directory(data.train_dir, batch_size=batch_size, target_size=data.target_size),
    samples_per_epoch=data.num_train,
    nb_epoch=nb_epoch,
    validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size, target_size=data.target_size),
    nb_val_samples=data.num_val,
    nb_worker=2,
    max_q_size=16)

# Save the model
generator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))

# Generate montage of sample-images
sample_size = 100
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=batch_size,
                                             target_size=data.target_size).next()
decoded_imgs = generator.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(net_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(net_name)))
