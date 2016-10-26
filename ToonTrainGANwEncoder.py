import gc
import os
import sys
import time

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Generator, Discriminator, GANwEncoder, Encoder
from constants import MODEL_DIR, IMG_DIR
from datasets import Imagenet
from utils import montage
import Queue as queue
import threading
import time


def disc_data(X, Y, Yd):
    Xd = np.concatenate((Y, Yd))
    yd = np.zeros((len(Y) + len(Yd), 1))
    yd[:len(Y)] = 1
    return Xd, yd


def generator_queue(generator, max_q_size=2, wait_time=0.05, nb_worker=1):
    q = queue.Queue()
    _stop = threading.Event()
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
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        raise

    return q, _stop


batch_size = 64
chunk_size = 32 * batch_size
nb_epoch = 1
r_weight = 40.0
num_train = 200000

# Get the data-set object
data = Imagenet(num_train=num_train, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, load_weights=True)
discriminator = Discriminator(data.dims, load_weights=False, train=True)  # TODO: Maybe change to load_weights
gan, gen_gan, disc_gan = GANwEncoder(data.dims, load_weights=True, recon_weight=r_weight)
encoder, _ = Encoder(data.dims, load_weights=True, train=False)

net_specs = 'rw{}'.format(r_weight)
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

# Create queue for training data
data_gen_queue, _stop = generator_queue(
    datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size))

# Training
print('Adversarial training...')
loss_avg_rate = 0.5
loss_target_ratio = 0.1
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    g_loss = None
    d_loss = None

    samples_seen = 0
    while samples_seen < num_train:

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                X_train, Y_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        if not d_loss or d_loss > g_loss * loss_target_ratio:
            print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
            # Reload the weights
            generator.load_weights(gen_weights)

            # Construct data for discriminator training
            Yd = generator.predict(X_train, batch_size=batch_size)
            Xd_train, yd_train = disc_data(X_train, Y_train, Yd)

            # Train discriminator
            discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)

            # Test discriminator
            Yd = generator.predict(X_test, batch_size=batch_size)
            Xd_test, yd_test = disc_data(X_test, Y_test, Yd)
            d_loss = discriminator.evaluate(Xd_test, yd_test, batch_size=batch_size, verbose=0)

            # Record and print loss
            losses["d"].append(d_loss)
            print('d-Loss: {}'.format(d_loss))

            # Save the weights
            discriminator.save_weights(disc_weights)
            del Xd_train, yd_train, Yd

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        disc_gan.load_weights(disc_weights)

        for i in range(2):
            # Train generator
            Yg_train = encoder.predict(Y_train)
            yg_train = np.ones((len(Y_train), 1))
            h = gan.fit(x=X_train, y=[yg_train, Yg_train, Y_train], nb_epoch=1, batch_size=batch_size, verbose=0)

            # Test generator
            Yg_test = encoder.predict(Y_test)
            yg_test = np.ones((len(Y_test), 1))
            res = gan.evaluate(X_test, [yg_train, Yg_test, Y_test], batch_size=batch_size, verbose=0)
            g_loss = res[0]
            e_loss = res[1]
            r_loss = res[2]

            # Record and print loss
            losses["g"].append(g_loss)
            print('g-Loss: {} r-Loss: {} e-Loss: {}'.format(g_loss, r_loss, e_loss))

            if g_loss * loss_target_ratio < d_loss:
                break

        # Save the weights
        gen_gan.save_weights(gen_weights)

        # Generate montage of test-images
        if not chunk % 5:
            generator.load_weights(gen_weights)
            decoded_imgs = generator.predict(X_test[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:18, :, :] * 0.5 + 0.5, X_test[:18] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, '{}-Epoch:{}-Chunk:{}.jpeg'.format(gen_name, epoch, chunk)))

        chunk += 1
        samples_seen += chunk_size

        sys.stdout.flush()
        del X_train, Y_train
        gc.collect()

disc_gan.save_weights(disc_weights)
gen_gan.save_weights(gen_weights)
