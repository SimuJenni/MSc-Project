import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import GAN, Disc2, gen_data, disc_data
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon, TinyImagenetToon
from utils import montage, generator_queue
import keras.backend as K

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.75, 1.0],
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
noise = K.variable(value=0.5, name='sigma')
noise_decay_rate = 0.95
merge_order = K.variable(value=np.int16(0), name='merge_order', dtype='int16')

# Load the models
discriminator = Disc2(data.dims, load_weights=False, num_layers=num_layers, noise=noise)
GAN, gen_gan, disc_gan = GAN(data.dims,
                             order=merge_order,
                             batch_size=batch_size,
                             num_layers=num_layers,
                             load_weights=load_weights,
                             noise=noise,
                             train_disc=False)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}_{}_out.hdf5'.format(gen_gan.name, data.name))
disc_weights = os.path.join(MODEL_DIR, '{}_{}_out.hdf5'.format(disc_gan.name, data.name))
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
    count_skip = 0

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

        if train_disc or l1 < 1.2 or d_loss>l1 or count_skip > 0:
            # Train discriminator
            print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
            # Xd_train, yd_train = disc_data(toon_train, img_train, im_pred)
            # Prepare training data
            im_pred = gen_gan.predict(gen_data(toon_train, edge_train), batch_size=batch_size)
            y = np.random.choice([0.0, 1.0], size=(len(img_train), 1))
            y_ind = np.where(y > 0.5)[0]
            y2 = np.zeros((len(img_train), 4, 4, 1))
            y2[y_ind, :] = 1.0
            X = np.concatenate((im_pred, img_train), axis=3)
            X[y_ind, :] = np.concatenate((img_train[y_ind, :], im_pred[y_ind, :]), axis=3)

            h = discriminator.fit(X, [y, y2], nb_epoch=1, batch_size=batch_size, verbose=0)
            d_loss = h.history['loss']
            print('D-Loss: {}'.format(d_loss))

            # Update the weights
            disc_gan.set_weights(discriminator.get_weights())
            train_disc = False # TODO: remove?
            count_skip = 0
        else:
            count_skip += 1

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Train generator
        if chunk % 2 == 0:
            y = [np.ones((len(toon_train), 1)), np.ones((len(toon_train), 4, 4, 1)), img_train]
        else:
            y = [np.zeros((len(toon_train), 1)), np.zeros((len(toon_train), 4, 4, 1)), img_train]

        h = GAN.fit(x=[gen_data(toon_train, edge_train), img_train], y=y,
                    nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        l1 = h.history['{}_loss_1'.format(GAN.output_names[0])][0]
        l2 = h.history['{}_loss_2'.format(GAN.output_names[1])][0]
        l3 = h.history['{}_loss'.format(GAN.output_names[2])][0]
        print('Loss: {} L_1: {} L_2: {} l3: {}'.format(t_loss, l1, l2, l3))

        # Generate montage of test-images
        if not chunk % 25:
            decoded_imgs = gen_gan.predict(gen_data(toon_test[:batch_size], edge_test[:batch_size]),
                                             batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, '{}-{}-Epoch:{}-Chunk:{}-big_m_bigF.jpeg'.format(GAN.name, data.name, epoch, chunk)))
            # Save the weights
            disc_gan.save_weights(disc_weights)
            gen_gan.save_weights(gen_weights)
            del toon_train, img_train, h, decoded_imgs
            gc.collect()
        sys.stdout.flush()
        K.set_value(merge_order, np.int16(chunk % 2))

    # Save the weights
    disc_gan.save_weights(disc_weights)
    gen_gan.save_weights(gen_weights)

    # # Update noise lvl
    # if noise:
    #     new_sigma = K.get_value(noise) * noise_decay_rate
    #     print('Lowering noise-level to {}'.format(new_sigma))
    #     K.set_value(noise, new_sigma)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
