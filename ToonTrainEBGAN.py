import gc
import os
import sys
import time

import numpy as np

from ToonDataGenerator import ImageDataGenerator, generator_queue
from ToonNet import EBGAN, Generator
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator()

# Training parameters
num_layers = 3
batch_size = 200
chunk_size = 10 * batch_size
num_chunks = data.num_train // chunk_size
nb_epoch = 20
r_weight = 10.0

# Load the models
generator = Generator(input_shape=data.dims, num_layers=num_layers, batch_size=batch_size)
dGAN, d_gen, d_disc = EBGAN(data.dims, batch_size=batch_size, load_weights=True, train_disc=True,
                            num_layers_d=num_layers,
                            num_layers_g=num_layers,
                            r_weight=r_weight)
gGAN, g_gen, g_disc = EBGAN(data.dims, batch_size=batch_size, load_weights=True, train_disc=False,
                            num_layers_d=num_layers,
                            num_layers_g=num_layers,
                            r_weight=r_weight)
gGAN.summary()

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(gGAN.name))
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(dGAN.name))

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size, target_size=data.target_size).next()
montage(X_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(gGAN.name)))
montage(Y_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(gGAN.name)))

# Training
print('EBGAN training: {}'.format(gGAN.name))

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
        max_q_size=8,
        nb_worker=2)

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                X_train, Y_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        target = np.zeros_like(Y_train)
        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        # Update the weights
        d_disc.set_weights(g_disc.get_weights())
        d_gen.set_weights(g_gen.get_weights())

        # Train discriminator
        h = dGAN.fit(x=[X_train, Y_train], y=[target, target, target], nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        l1 = h.history['{}_loss'.format(dGAN.output_names[0])][0]
        l2 = h.history['{}_loss'.format(dGAN.output_names[1])][0]
        l3 = h.history['{}_loss'.format(dGAN.output_names[2])][0]

        # Record and print loss
        print('Loss: {} L_1: {} L_2: {} L_3: {}'.format(t_loss, l1, l2, l3))

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Update the weights
        g_disc.set_weights(d_disc.get_weights())
        g_gen.set_weights(d_gen.get_weights())

        # Train generator
        h = gGAN.fit(x=[X_train, Y_train], y=[target, target], nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        l2 = h.history['{}_loss'.format(gGAN.output_names[0])][0]
        l3 = h.history['{}_loss'.format(gGAN.output_names[1])][0]

        # Record and print loss
        print('Loss: {} L_1:{} L_2: {} L_3: {}'.format(t_loss+l1, l1, l2, l3))

        # Generate montage of test-images
        if not chunk % 50:
            generator.set_weights(g_gen.get_weights())
            decoded_imgs = generator.predict(X_test[:batch_size], batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, '{}-Epoch:{}-Chunk:{}.jpeg'.format(gGAN.name, epoch, chunk)))
            # Save the weights
            g_disc.save_weights(disc_weights)
            g_gen.save_weights(gen_weights)
            del X_train, Y_train, target, h, decoded_imgs
            gc.collect()
        sys.stdout.flush()


    # Save the weights
    g_disc.save_weights(disc_weights)
    g_gen.save_weights(gen_weights)

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
