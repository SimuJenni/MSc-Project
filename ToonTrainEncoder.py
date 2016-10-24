import os

from DataGenerator import ImageDataGenerator, Y2X_Y2Y
from ToonNet import Encoder
from constants import MODEL_DIR
from datasets import Imagenet

batch_size = 64
num_epochs = 2
samples_per_epoch = 200000

# Get the data-set object
data = Imagenet(num_train=samples_per_epoch, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
encoder, generator = Encoder(data.dims, load_weights=False, train=True)
print('Training encoder...')

# Training
history = generator.fit_generator(
    datagen.flow_from_directory(data.train_dir, batch_size=batch_size, xy_fun=Y2X_Y2Y, target_size=data.target_size),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=num_epochs,
    validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size,
                                                xy_fun=Y2X_Y2Y, target_size=data.target_size),
    nb_val_samples=20000,
    nb_worker=2)

encoder.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))
generator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
