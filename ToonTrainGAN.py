import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import ToonGAN, Gen, gen_data
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
batch_size = 100
chunk_size = 4 * batch_size
num_chunks = data.num_train // chunk_size
nb_epoch = 100
load_weights = False

# Load the models
generator = Gen(input_shape=data.dims, num_layers=num_layers, batch_size=batch_size)
dGAN, d_gen, d_disc = ToonGAN(data.dims,
                              batch_size=batch_size,
                              num_layers=num_layers,
                              load_weights=load_weights,
                              train_disc=True)
gGAN, g_gen, g_disc = ToonGAN(data.dims,
                              batch_size=batch_size,
                              num_layers=num_layers,
                              load_weights=load_weights,
                              train_disc=False)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(gGAN.name))
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(dGAN.name))
# g_disc.load_weights(disc_weights)
# g_gen.load_weights(gen_weights)

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()
montage(toon_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-toon.jpeg'.format(gGAN.name, data.name)))
montage(np.squeeze(edge_test[:100]), os.path.join(IMG_DIR, '{}-{}-edge.jpeg'.format(gGAN.name, data.name)))
montage(im_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-images.jpeg'.format(gGAN.name, data.name)))

# Training
print('EBGAN training: {}'.format(gGAN.name))
l2 = 0

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
        # Update the weights
        d_disc.set_weights(g_disc.get_weights())
        d_gen.set_weights(g_gen.get_weights())

        # Train discriminator
        # h = dGAN.fit(x=[X_train, Y_train], y=[target]*len(dGAN.output_names), nb_epoch=1, batch_size=batch_size, verbose=0)
        h = dGAN.fit(x=[gen_data(toon_train, edge_train), img_train], y=[np.zeros((len(toon_train), 1))] * 2 + [target],
                     nb_epoch=1, batch_size=batch_size, verbose=0)
        for key, value in h.history.iteritems():
            print('{}: {}'.format(key, value))

        # t_loss = h.history['loss'][0]
        # l1 = h.history['{}_loss'.format(dGAN.output_names[0])][0]
        # l3 = h.history['{}_loss'.format(dGAN.output_names[1])][0]
        # l4 = h.history['{}_loss'.format(dGAN.output_names[2])][0]
        #
        # # Record and print loss
        # print('Loss: {} L_1: {} L_2: {} L_3: {} L_4: {}'.format(t_loss, l1, l2, l3, l4))

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Update the weights
        g_disc.set_weights(d_disc.get_weights())
        g_gen.set_weights(d_gen.get_weights())

        # Train generator

        # h = gGAN.fit(x=[X_train, Y_train], y=[target]*len(gGAN.output_names), nb_epoch=1, batch_size=batch_size, verbose=0)
        h = gGAN.fit(x=[gen_data(toon_train, edge_train), img_train],
                     y=[np.zeros((len(toon_train), 1)), target, target], nb_epoch=1, batch_size=batch_size, verbose=0)
        for key, value in h.history.iteritems():
            print('{}: {}'.format(key, value))

        # t_loss = h.history['loss'][0]
        # l2 = h.history['{}_loss'.format(gGAN.output_names[0])][0]
        # l3 = h.history['{}_loss'.format(gGAN.output_names[1])][0]
        #
        # # Record and print loss
        # print('Loss: {} L_1: {} L_2: {} L_3: {} L_4: {}'.format(t_loss, l1, l2, l3, l4))

        # Generate montage of test-images
        if not chunk % 25:
            generator.set_weights(g_gen.get_weights())
            decoded_imgs = generator.predict(gen_data(toon_test[:batch_size], edge_test[:batch_size]),
                                             batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, '{}-{}-Epoch:{}-Chunk:{}.jpeg'.format(gGAN.name, data.name, epoch, chunk)))
            # Save the weights
            g_disc.save_weights(disc_weights)
            g_gen.save_weights(gen_weights)
            del toon_train, img_train, target, h, decoded_imgs
            gc.collect()
        sys.stdout.flush()

    # Save the weights
    g_disc.save_weights(disc_weights)
    g_gen.save_weights(gen_weights)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
