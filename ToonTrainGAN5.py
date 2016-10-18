import gc
import os
import sys

import numpy as np
from keras.layers import Input, merge
from keras.models import Model, Sequential
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator
from constants import MODEL_DIR, IMG_DIR
from datasets.TinyImagenet import TinyImagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


batch_size = 32
chunk_size = 50 * batch_size
nb_epoch = 2

# Get the data-set object
data = TinyImagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size, bn_mode=0)
#toonAE.load_weights(os.path.join(MODEL_DIR, 'ToonAE2.hdf5'))

# Load the discriminator
toonDisc = ToonDiscriminator(input_shape=data.dims, bn_mode=0)

# Stick them together
make_trainable(toonDisc, False)

toonGAN = Sequential()
toonGAN.add(toonAE)
toonGAN.add(toonDisc)
#toonAE.compile(optimizer=opt, loss='mse')
toonGAN.compile(optimizer=opt, loss='binary_crossentropy')
make_trainable(toonDisc, True)
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')
toonGAN.summary()

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
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):
        if d_loss_avg > loss_target_ratio * g_loss_avg:
            # Construct data for discriminator training
            Y_pred = toonAE.predict(X_train)
            X = np.concatenate((Y_train, Y_pred))
            y = np.zeros((len(Y_train) + len(Y_pred), 1))
            y[:len(Y_train)] = 1

            # Train discriminator
            make_trainable(toonDisc, True)
            d_loss = toonDisc.train_on_batch(X, y)
            losses["d"].append(d_loss)
            d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
            del X, Y_pred
        else:
            # Train generator
            y = np.array([1] * len(Y_train))
            make_trainable(toonDisc, False)
            g_loss, r_loss = toonGAN.train_on_batch(X_train, y)
            losses["g"].append(g_loss)
            g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss

        # Generate montage of test-images
        if not chunk % 200:
            decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
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

    toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc_GAN4-Epoch:{}.hdf5'.format(epoch)))
    toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAE_GAN4-Epoch:{}.hdf5'.format(epoch)))

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN4.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN4.hdf5'))
