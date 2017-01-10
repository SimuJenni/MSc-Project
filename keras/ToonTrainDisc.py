import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import Disc2
from constants import MODEL_DIR
from keras.datasets import CIFAR10_Toon
from keras.utils import generator_queue

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
disc = Disc2(data.dims, num_layers=num_layers)

# Paths for storing the weights
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(disc.name))

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()

# Training
print('Discriminaotor training: {}'.format(disc.name))

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

        target = toon_train
        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))

        # Train discriminator
        y = np.random.choice([0.0, 1.0], size=(len(img_train), 1))
        y_ind = np.where(y > 0.5)[0]
        X = np.concatenate((target, img_train), axis=3)
        X[y_ind, :] = np.concatenate((img_train[y_ind, :], target[y_ind, :]), axis=3)
        h = disc.fit(x=X, y=y,
                     batch_size=batch_size, verbose=0, nb_epoch=1)
        print(h.history)

        sys.stdout.flush()

    # Save the weights
    disc.save_weights(disc_weights)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
