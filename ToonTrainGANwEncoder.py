import Queue as queue
import gc
import os
import sys
import threading
import time

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Generator, Discriminator, GANwEncoder, Encoder
from constants import MODEL_DIR, IMG_DIR
from datasets import Imagenet
from utils import montage


def disc_data(X, Y, Yd):
    Xd = np.concatenate((Y, Yd))
    yd = np.zeros((len(Y) + len(Yd), 1))
    yd[:len(Y)] = 1
    return Xd, yd


def generator_queue(generator, max_q_size=4, wait_time=0.05, nb_worker=2):
    q = queue.Queue()
    _stop = threading.Event()
    threads = []
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(nb_worker):
            thread = threading.Thread(target=data_generator_task)
            threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        raise

    return q, _stop, threads


batch_size = 64
chunk_size = 64 * batch_size
num_chunks = 100
nb_epoch = 4
r_weight = 50.0
e_weight = r_weight/100
num_train = num_chunks*chunk_size
num_res_g = 16

# Get the data-set object
data = Imagenet(num_train=num_train, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, load_weights=True, num_res=num_res_g)
discriminator = Discriminator(data.dims, load_weights=True, train=True)  # TODO: Maybe change to load_weights
gan, gen_gan, disc_gan = GANwEncoder(data.dims, load_weights=True, recon_weight=r_weight, enc_weight=e_weight,
                                     num_res_g=num_res_g)
encoder, _ = Encoder(data.dims, load_weights=True, train=False)

net_specs = 'rw{}_ew{}'.format(r_weight, e_weight)
gen_name = '{}_{}'.format(gen_gan.name, net_specs)
disc_name = '{}_{}'.format(disc_gan.name, net_specs)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(gen_name))
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(disc_name))
generator.save_weights(gen_weights)
discriminator.save_weights(disc_weights)

# Store losses
losses = {"d": [], "g": []}

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size, target_size=data.target_size).next()

# Training
print('Adversarial training: {}'.format(gen_name))

loss_target_ratio = 0.1
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    g_loss = None
    d_loss = None

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size))

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                X_train, Y_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        load_disc = False

        if not d_loss or d_loss > g_loss * loss_target_ratio:
            print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
            # Reload the weights
            generator.load_weights(gen_weights)

            # Construct data for discriminator training
            Yd = generator.predict(X_train, batch_size=batch_size)
            Xd_train, yd_train = disc_data(X_train, Y_train, Yd)

            # Train discriminator
            h = discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)
            d_loss = h.history['loss'][0]

            # Record and print loss
            losses["d"].append(d_loss)
            print('d-Loss: {}'.format(d_loss))

            # Save the weights
            discriminator.save_weights(disc_weights)
            load_disc = True
            del Xd_train, yd_train, Yd

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        if load_disc:
            disc_gan.load_weights(disc_weights)

        # Train generator
        Yg_train = encoder.predict(Y_train)
        yg_train = np.ones((len(Y_train), 1))
        h = gan.fit(x=X_train, y=[yg_train, Yg_train, Y_train], nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        g_loss = h.history['{}_loss'.format(gan.output_names[0])][0]
        e_loss = h.history['{}_loss'.format(gan.output_names[1])][0]
        r_loss = h.history['{}_loss'.format(gan.output_names[2])][0]

        # Record and print loss
        losses["g"].append(g_loss)
        print('Loss: {} g-Loss: {} r-Loss: {} e-Loss: {}'.format(t_loss, g_loss, r_loss, e_loss))

        if g_loss * loss_target_ratio < d_loss:
            break

        # Save the weights
        gen_gan.save_weights(gen_weights)

        # Generate montage of test-images
        if not chunk % 10:
            generator.load_weights(gen_weights)
            decoded_imgs = generator.predict(X_test[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:18, :, :] * 0.5 + 0.5, X_test[:18] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, '{}-Epoch:{}-Chunk:{}.jpeg'.format(gen_name, epoch, chunk)))

        sys.stdout.flush()
        del X_train, Y_train
        gc.collect()

    _stop.set()

disc_gan.save_weights(disc_weights)
gen_gan.save_weights(gen_weights)
