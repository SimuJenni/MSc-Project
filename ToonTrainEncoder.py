import os

from DataGenerator import ImageDataGenerator, Y2X_Y2Y
from ToonNet import Encoder
from constants import MODEL_DIR, IMG_DIR
from datasets import Imagenet
from utils import montage

batch_size = 64
num_epochs = 2
samples_per_epoch = 200000

# Get the data-set object
data = Imagenet(num_train=samples_per_epoch, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
encoder, generator = Encoder(data.dims, load_weights=True, train=True)  #TODO: Remove load_weights later
gen_name = '{}-{}'.format(generator.name, data.name)
enc_name = '{}-{}'.format(encoder.name, data.name)
print('Training network: {}'.format(enc_name))

# Training
history = generator.fit_generator(
    datagen.flow_from_directory(data.train_dir, batch_size=batch_size, xy_fun=Y2X_Y2Y, target_size=data.target_size),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=num_epochs,
    validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size,
                                                xy_fun=Y2X_Y2Y, target_size=data.target_size),
    nb_val_samples=20000,
    nb_worker=2)

# Save the weights
encoder.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(enc_name)))
generator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(gen_name)))

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size, target_size=data.target_size, xy_fun=Y2X_Y2Y).next()
decoded_imgs = generator.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(gen_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(gen_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(gen_name)))