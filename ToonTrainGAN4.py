import gc
import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import GenAndDisc, Gan
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from datasets.TinyImagenet import TinyImagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


batch_size = 32
chunk_size = 50 * batch_size
nb_epoch = 2

# Get the data-set object
data = TinyImagenet()
datagen = ImageDataGenerator()

# Store losses
losses = {"d": [], "g": []}

# Training
print('Adversarial training...')
loss_avg_rate = 0.5
loss_target_ratio = 0.5
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    g_loss_avg = 3
    d_loss_avg = 3
    r_loss = 100
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size, target_size=(64, 64)):
        if d_loss_avg > loss_target_ratio * g_loss_avg:
            generator, discriminator = GenAndDisc(data.dims, batch_size)

            # Construct data for discriminator training
            Y_pred = generator.predict(X_train)
            X = np.concatenate((Y_train, Y_pred))
            y = np.zeros((len(Y_train) + len(Y_pred), 1))
            y[:len(Y_train)] = 1

            # Train discriminator
            make_trainable(discriminator, True)
            d_loss = discriminator.train_on_batch(X, y)
            losses["d"].append(d_loss)
            d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
            del X, Y_pred
        else:
            gan, generator, discriminator = Gan(data.dims, batch_size)
            # Train generator
            y = np.array([1] * len(Y_train))
            g_loss, r_loss = gan.train_on_batch(X_train, y)
            losses["g"].append(g_loss)
            g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss

        # Generate montage of test-images
        if not chunk % 200:
            decoded_imgs = generator.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:12, :, :] * 0.5 + 0.5, X_train[:12] * 0.5 + 0.5, Y_train[:12] * 0.5 + 0.5)),
                os.path.join(IMG_DIR, 'GAN4-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))
        chunk += 1

        print('GAN4 Epoch: {}/{} Batch: {} Disc-Loss: {} Gen-Loss: {} Recon-Loss: {}'.format(epoch,
                                                                                             nb_epoch,
                                                                                             chunk,
                                                                                             d_loss_avg,
                                                                                             g_loss_avg,
                                                                                             r_loss))
        sys.stdout.flush()
        del X_train, Y_train, y
        gc.collect()

    discriminator.save_weights(os.path.join(MODEL_DIR, 'ToonDisc_GAN4-Epoch:{}.hdf5'.format(epoch)))
    generator.save_weights(os.path.join(MODEL_DIR, 'ToonAE_GAN4-Epoch:{}.hdf5'.format(epoch)))

discriminator.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN4.hdf5'))
generator.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN4.hdf5'))
