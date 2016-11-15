import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import GAN
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage, generator_queue

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

# Training parameters
num_layers = 3
batch_size = 200
chunk_size = 4 * batch_size
num_chunks = data.num_train // chunk_size
nb_epoch = 100
load_weights = False

# Load the models
dGAN, d_gen, d_disc = GAN(data.dims,
                              batch_size=batch_size,
                              num_layers=num_layers,
                              load_weights=load_weights,
                              train_disc=True)

# Paths for storing the weights
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(d_disc.name))

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()

# Training
print('Discriminaotor training: {}'.format(d_disc.name))

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
        max_q_size=32,
        nb_worker=8)

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                toon_train, edge_train, img_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        target = np.zeros_like(img_train)
        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))

        # Train discriminator
        h = dGAN.fit(x=[gen_data(toon_train, edge_train), img_train], y=[np.zeros((len(toon_train), 1))] * 2 + [target],
                     nb_epoch=1, batch_size=batch_size, verbose=0)
        for key, value in h.history.iteritems():
            print('{}: {}'.format(key, value))

        sys.stdout.flush()

    # Save the weights
    d_disc.save_weights(disc_weights)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
