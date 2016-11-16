import gc
import os
import sys
import time

import keras.backend as K
import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import GAN, gen_data
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
nb_epoch = 30
load_weights = False
noise = K.variable(value=0.25, name='sigma')
noise_decay_rate = 0.95

# Load the models
dGAN, gen_dgan, disc_dgan = GAN(data.dims,
                                batch_size=batch_size,
                                num_layers=num_layers,
                                load_weights=load_weights,
                                noise=noise,
                                train_disc=True)
gGAN, gen_ggan, disc_ggan = GAN(data.dims,
                                batch_size=batch_size,
                                num_layers=num_layers,
                                load_weights=load_weights,
                                noise=noise,
                                train_disc=False)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}_gan.hdf5'.format(gen_ggan.name))
disc_weights = os.path.join(MODEL_DIR, '{}_gan.hdf5'.format(disc_dgan.name))

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()
montage(toon_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-toon.jpeg'.format(gGAN.name, data.name)))
montage(np.squeeze(edge_test[:100]), os.path.join(IMG_DIR, '{}-{}-edge.jpeg'.format(gGAN.name, data.name)), gray=True)
montage(im_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-images.jpeg'.format(gGAN.name, data.name)))

# Training
print('GAN training: {}'.format(dGAN.name))

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    train_disc = True

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

        # Train discriminator
        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        h = dGAN.fit(x=[gen_data(toon_train, edge_train), img_train],
                     y=np.ones((len(toon_train), 1)),
                     nb_epoch=1, batch_size=batch_size, verbose=0)
        print(h.history)

        # Update the weights
        disc_ggan.set_weights(disc_dgan.get_weights())

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Train generator
        h = gGAN.fit(x=[gen_data(toon_train, edge_train), img_train],
                     y=[np.zeros((len(toon_train), 1)), img_train],
                     nb_epoch=1, batch_size=batch_size, verbose=0)
        print(h.history)

        # Generate montage of test-images
        if not chunk % 25:
            decoded_imgs = gen_ggan.predict(gen_data(toon_test[:batch_size], edge_test[:batch_size]),
                                            batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR,
                                 '{}-{}-Epoch:{}-Chunk:{}-big_m_bigF.jpeg'.format(gGAN.name, data.name, epoch, chunk)))
            # Save the weights
            disc_dgan.save_weights(disc_weights)
            gen_ggan.save_weights(gen_weights)
            del toon_train, img_train, h, decoded_imgs
            gc.collect()
        sys.stdout.flush()

    # Save the weights
    disc_dgan.save_weights(disc_weights)
    gen_ggan.save_weights(gen_weights)

    # Update noise lvl
    if noise:
        new_sigma = K.get_value(noise) * noise_decay_rate
        print('Lowering noise-level to {}'.format(new_sigma))
        K.set_value(noise, new_sigma)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
