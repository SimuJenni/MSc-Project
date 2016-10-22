import gc
import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Generator, DiscLwise, GanLwise
from constants import MODEL_DIR, IMG_DIR
from datasets import Imagenet
from utils import montage

batch_size = 32
chunk_size = 128 * batch_size
nb_epoch = 1
f_dims = [64, 96, 160, 256, 512]
r_weight = 100.0
layer = 4

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, batch_size, load_weights=True, f_dims=f_dims, resize_conv=True, w_outter=True)
discriminator = DiscLwise(data.dims, load_weights=True, f_dims=f_dims, train=True)
disc_gentrain = DiscLwise(data.dims, load_weights=True, f_dims=f_dims, train=False, layer=layer)
gan, gen_gan, disc_gan = GanLwise(data.dims, batch_size, load_weights=True, f_dims=f_dims, resize_conv=True,
                                  w_outter=True, recon_weight=r_weight, layer=layer)
net_specs = 'rw{}l{}'.format(r_weight, layer)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, 'gen_lwise_{}.hdf5'.format(net_specs))
disc_weights = os.path.join(MODEL_DIR, 'disc_lwise_{}.hdf5'.format(net_specs))
generator.save_weights(gen_weights)
discriminator.save_weights(disc_weights)

# Store losses
losses = {"d": [], "g": []}

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size).next()

# Training
print('Adversarial training...')
loss_avg_rate = 0.5
loss_target_ratio = 0.20
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    g_loss_avg = 6
    d_loss_avg = 6
    r_loss = 100
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):

        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        generator.load_weights(gen_weights)

        # Construct data for discriminator training
        Yd_train = generator.predict(X_train, batch_size=batch_size)
        Xd_train = np.concatenate((Y_train, Yd_train))
        yd_train = np.zeros((len(Y_train) + len(Yd_train), 1))
        yd_train[:len(Y_train)] = 1

        for i in range(2):
            # Train discriminator
            discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)

            # Test discriminator
            Yd_test = generator.predict(X_test, batch_size=batch_size)
            Xd_test = np.concatenate((Y_test, Yd_test))
            yd_test = np.zeros((len(Y_test) + len(Yd_test), 1))
            yd_test[:len(Y_test)] = 1
            d_loss = discriminator.evaluate(Xd_test, yd_test, batch_size=batch_size, verbose=0)

            # Record and print loss
            losses["d"].append(d_loss)
            d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
            print('d-Loss: {} d-Loss-avg: {}'.format(d_loss, d_loss_avg))

            if d_loss_avg < g_loss_avg * loss_target_ratio:
                break

        # Save the weights
        discriminator.save_weights(disc_weights)

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        disc_gan.load_weights(disc_weights)
        disc_gentrain.load_weights(disc_weights)

        for i in range(4):
            # Train generator
            Yg_train = disc_gentrain.predict(Y_train)
            Yg_train[-1] = np.ones((len(Y_train), 1))
            gan.fit(x=X_train, y=Yg_train + [Y_train], nb_epoch=1, batch_size=batch_size, verbose=0)

            # Test generator
            Yg_test = disc_gentrain.predict(Y_train)
            Yg_test[-1] = np.ones((len(Y_test), 1))
            res = gan.evaluate(X_test, Yg_test + [Y_test], batch_size=batch_size, verbose=0)
            g_loss = res[-2]
            r_loss = res[-1]

            # Record and print loss
            losses["g"].append(g_loss)
            g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss
            print('g-Loss: {} g-Loss-avg: {} r-Loss: {}'.format(g_loss, g_loss_avg, r_loss))

            if g_loss_avg * loss_target_ratio < d_loss_avg:
                break

        # Save the weights
        gen_gan.save_weights(gen_weights)

        # Generate montage of test-images
        if not chunk % 2:
            generator.load_weights(gen_weights)
            decoded_imgs = generator.predict(X_test[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:12, :, :] * 0.5 + 0.5, X_test[:12] * 0.5 + 0.5, Y_test[:12] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, 'GANlwise-Epoch:{}-Chunk:{}-Spec:{}.jpeg'.format(epoch, chunk, net_specs)))
        chunk += 1

        sys.stdout.flush()
        del X_train, Y_train, Yd_train, Xd_train, yd_train
        gc.collect()

disc_gan.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGANlwise_{}.hdf5'.format(net_specs)))
gen_gan.save_weights(os.path.join(MODEL_DIR, 'ToonAEGANlwise_{}.hdf5'.format(net_specs)))
