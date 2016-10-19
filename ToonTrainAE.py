import os

from keras.callbacks import ModelCheckpoint, TensorBoard

from DataGenerator import ImageDataGenerator
from ToonNet import Generator
from constants import MODEL_DIR, IMG_DIR, LOG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 24
nb_epoch = 1
samples_per_epoch = 500000

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
generator = Generator(input_shape=data.dims, batch_size=batch_size, load_weights=False)
generator.summary()

# Name used for saving of model and outputs
net_name = '{}-Data:{}'.format('ToonAE', data.name)
print('Training network: {}'.format(net_name))

# Training
history = generator.fit_generator(datagen.flow_from_directory(data.train_dir, batch_size=batch_size),
                                  samples_per_epoch=samples_per_epoch,
                                  nb_epoch=nb_epoch,
                                  validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size),
                                  nb_val_samples=32000,
                                  nb_worker=4,
                                  callbacks=[ModelCheckpoint(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name))),
                                             TensorBoard(log_dir=LOG_DIR, histogram_freq=1)])

# Save the model
generator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size).next()
decoded_imgs = generator.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(net_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(net_name)))
