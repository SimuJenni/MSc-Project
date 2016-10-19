import gc
import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import GenAndDisc, Gan
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


batch_size = 32
chunk_size = 300 * batch_size
nb_epoch = 1

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
generator, discriminator = GenAndDisc(data.dims, batch_size, load_weights=True)
gan, gen_gan, disc_gan = Gan(data.dims, batch_size, load_weights=True)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, 'gen.hdf5')
disc_weights = os.path.join(MODEL_DIR, 'disc.hdf5')
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
    g_loss_avg = 1
    d_loss_avg = 1
    r_loss = 100
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        gen_gan.load_weights(gen_weights)
        disc_gan.load_weights(disc_weights)

        # Train generator
        y = np.ones((len(Y_train), 1))
        gan.fit(X_train, y, nb_epoch=1, batch_size=batch_size)

        # Test generator
        y = np.ones(len(X_test), 1)
        g_loss, r_loss = gan.evaluate(X_test, y, batch_size=batch_size, verbose=0)

        # Record and print loss
        losses["g"].append(g_loss)
        g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss
        print('g-Loss: {} r-Loss'.format(g_loss, r_loss))

        # Save the weights
        gen_gan.save_weights(gen_weights)
        disc_gan.save_weights(disc_weights)

        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        # Reload the weights
        generator.load_weights(gen_weights)
        discriminator.load_weights(disc_weights)

        # Construct data for discriminator training
        Y_pred = generator.predict(X_train, batch_size=batch_size)
        X = np.concatenate((Y_train, Y_pred))
        y = np.zeros((len(Y_train) + len(Y_pred), 1))
        y[:len(Y_train)] = 1

        # Train discriminator
        make_trainable(discriminator, True)
        discriminator.fit(X, y, nb_epoch=1, batch_size=batch_size)

        # Test discriminator
        Y_pred = generator.predict(X_test, batch_size=batch_size)
        X = np.concatenate((Y_test, Y_pred))
        y = np.zeros((len(Y_test) + len(Y_pred), 1))
        y[:len(Y_test)] = 1
        d_loss = discriminator.evaluate(X, y, batch_size=batch_size, verbose=0)

        # Record and print loss
        losses["d"].append(d_loss)
        d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
        del X, Y_pred
        print('d-Loss: {}'.format(d_loss))

        # Save the weights
        generator.save_weights(gen_weights)
        discriminator.save_weights(disc_weights)

        # Generate montage of test-images
        if not chunk % 2:
            generator.load_weights(gen_weights)
            decoded_imgs = generator.predict(X_test[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:12, :, :] * 0.5 + 0.5, X_test[:12] * 0.5 + 0.5, Y_test[:12] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))
        chunk += 1

        sys.stdout.flush()
        del X_train, Y_train, y
        gc.collect()


disc_gan.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN4.hdf5'))
gen_gan.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN4.hdf5'))
