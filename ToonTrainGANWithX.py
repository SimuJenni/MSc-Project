import gc
import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Generator, DiscriminatorWithX, GanWithX
from constants import MODEL_DIR, IMG_DIR
from datasets import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


batch_size = 16
chunk_size = 100 * batch_size
nb_epoch = 1
f_dims = [64, 128, 256, 512, 1024]

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, batch_size, load_weights=True, f_dims=f_dims)
discriminator = DiscriminatorWithX(data.dims, load_weights=True, f_dims=f_dims)
gan, gen_gan, disc_gan = GanWithX(data.dims, batch_size, load_weights=True, f_dims=f_dims, l2_rate=100.0)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, 'gen_wx100.hdf5')
disc_weights = os.path.join(MODEL_DIR, 'disc_wx100.hdf5')
generator.save_weights(gen_weights)
discriminator.save_weights(disc_weights)

# Store losses
losses = {"d": [], "g": []}

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size).next()

# Training
print('Adversarial training...')
loss_avg_rate = 0.5
loss_target_ratio = 0.5
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    loss_ratio = 2
    g_loss_avg = 6
    d_loss_avg = 6
    r_loss = 100
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):

        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        generator.load_weights(gen_weights)
        discriminator.load_weights(disc_weights)

        # Construct data for discriminator training
        Yd_train = generator.predict(X_train, batch_size=batch_size)
        Xd_train = np.concatenate((np.concatenate((X_train, Y_train), axis=3), np.concatenate((X_train, Yd_train), axis=3)))
        yd_train = np.zeros((len(Y_train) + len(Yd_train), 1))
        yd_train[:len(Y_train)] = 1

        while True:
            # Train discriminator
            make_trainable(discriminator, True)
            discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size)

            # Test discriminator
            Yd_test = generator.predict(X_test, batch_size=batch_size)
            Xd_test = np.concatenate((np.concatenate((X_train, Y_test), axis=3), np.concatenate((X_train, Yd_test), axis=3)))
            yd_test = np.zeros((len(Y_test) + len(Yd_test), 1))
            yd_test[:len(Y_test)] = 1
            d_loss = discriminator.evaluate(Xd_test, yd_test, batch_size=batch_size, verbose=0)

            # Record and print loss
            losses["d"].append(d_loss)
            d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
            print('d-Loss: {} d-Loss-avg: {}'.format(d_loss, d_loss_avg))

            if g_loss_avg / d_loss_avg > loss_ratio:
                break

        # Save the weights
        generator.save_weights(gen_weights)
        discriminator.save_weights(disc_weights)

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        gen_gan.load_weights(gen_weights)
        disc_gan.load_weights(disc_weights)

        for i in range(2):
            # Train generator
            yg_train = np.ones((len(Y_train), 1))
            gan.fit(x=X_train, y=[yg_train, Y_train], nb_epoch=1, batch_size=batch_size)

            # Test generator
            yg_test = np.ones((len(X_test), 1))
            res = gan.evaluate(X_test, [yg_test, Y_test], batch_size=batch_size, verbose=0)
            g_loss = res[1]
            r_loss = res[2]

            # Record and print loss
            losses["g"].append(g_loss)
            g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss
            print('g-Loss: {} g-Loss-avg: {} r-Loss: {}'.format(g_loss, g_loss_avg, r_loss))

            if g_loss_avg / d_loss_avg < loss_ratio:
                break

        # Save the weights
        gen_gan.save_weights(gen_weights)
        disc_gan.save_weights(disc_weights)

        # Generate montage of test-images
        if not chunk % 2:
            generator.load_weights(gen_weights)
            decoded_imgs = generator.predict(X_test[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:12, :, :] * 0.5 + 0.5, X_test[:12] * 0.5 + 0.5, Y_test[:12] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, 'GANnorm-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))
        chunk += 1

        sys.stdout.flush()
        del X_train, Y_train, Yd_train, Xd_train, yd_train
        gc.collect()

disc_gan.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGANwX100.hdf5'))
gen_gan.save_weights(os.path.join(MODEL_DIR, 'ToonAEGANwX100.hdf5'))
