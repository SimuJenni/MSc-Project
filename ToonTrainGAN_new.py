import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import GAN, Gen, Disc, gen_data, disc_data
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage, generator_queue
import keras.backend as K

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
noise = K.variable(value=0.2, name='sigma')
noise_decay_rate = 0.9

# Load the models
generator = Gen(input_shape=data.dims, num_layers=num_layers, batch_size=batch_size)
discriminator = Disc(data.dims, load_weights=False, num_layers=num_layers, noise=noise)
GAN, gen_gan, disc_gan = GAN(data.dims,
                             batch_size=batch_size,
                             num_layers=num_layers,
                             load_weights=load_weights,
                             noise=noise)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}_gan.hdf5'.format(gen_gan.name))
disc_weights = os.path.join(MODEL_DIR, '{}_gan.hdf5'.format(disc_gan.name))
disc_gan.set_weights(discriminator.get_weights())

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()
montage(toon_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-toon.jpeg'.format(GAN.name, data.name)))
montage(np.squeeze(edge_test[:100]), os.path.join(IMG_DIR, '{}-{}-edge.jpeg'.format(GAN.name, data.name)), gray=True)
montage(im_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-images.jpeg'.format(GAN.name, data.name)))

# Training
print('GAN training: {}'.format(GAN.name))

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

        target = np.zeros_like(img_train)

        # Update the weights
        generator.set_weights(gen_gan.get_weights())

        # Prepare training data
        im_pred = generator.predict(gen_data(toon_train, edge_train), batch_size=batch_size)
        d_out, dp_out = disc_gan.predict(im_pred, batch_size=batch_size)
        print(-np.mean(np.log(1.0-d_out)))
        if -np.mean(np.log(1.0-d_out)) < 1.0 or train_disc:
            # Train discriminator
            print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
            Xd_train, yd_train = disc_data(toon_train, img_train, im_pred)

            h = discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)
            d_loss = h.history['loss']
            print('D-Loss: {}'.format(d_loss))

            # Update the weights
            disc_gan.set_weights(discriminator.get_weights())
            train_disc = False

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Train generator
        h = GAN.fit(x=[gen_data(toon_train, edge_train), img_train],
                    y=[np.ones((len(toon_train), 1)), img_train, dp_out],
                    nb_epoch=1, batch_size=batch_size, verbose=0)

        loss_str = ''
        for key, value in sorted(h.history.iteritems()):
            loss_str = '{}{}: {} '.format(loss_str, key, value)
        print(loss_str)

        # Generate montage of test-images
        if not chunk % 25:
            generator.set_weights(gen_gan.get_weights())
            decoded_imgs = generator.predict(gen_data(toon_test[:batch_size], edge_test[:batch_size]),
                                             batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, '{}-{}-Epoch:{}-Chunk:{}-big_m_bigF.jpeg'.format(GAN.name, data.name, epoch, chunk)))
            # Save the weights
            disc_gan.save_weights(disc_weights)
            gen_gan.save_weights(gen_weights)
            del toon_train, img_train, target, h, decoded_imgs, im_pred
            gc.collect()
        sys.stdout.flush()

    # Save the weights
    disc_gan.save_weights(disc_weights)
    gen_gan.save_weights(gen_weights)

    # Update noise lvl
    if noise:
        new_sigma = K.get_value(noise) * noise_decay_rate
        print('Lowering noise-level to {}'.format(new_sigma))
        K.set_value(noise, new_sigma)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
