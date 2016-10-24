import os

from DataGenerator import ImageDataGenerator, Y2X_Y2Y
from ToonNet import Encoder
from constants import MODEL_DIR
from datasets import Imagenet

batch_size = 64
chunk_size = 50 * batch_size
num_epochs = 1
samples_per_epoch = 1152000

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
encoder, generator = Encoder(data.dims, load_weights=False, train=True)
print('Training encoder...')

# Training
history = generator.fit_generator(datagen.flow_from_directory(data.train_dir, batch_size=batch_size, xy_fun=Y2X_Y2Y),
                                  samples_per_epoch=samples_per_epoch,
                                  nb_epoch=num_epochs,
                                  validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size,
                                                                              xy_fun=Y2X_Y2Y),
                                  nb_val_samples=32000,
                                  nb_worker=4)

encoder.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))
generator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(generator.name)))
