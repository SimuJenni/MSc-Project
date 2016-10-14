import gc
import os
import sys

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator3
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def compute_accuracy(y_hat, y):
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot
    return acc


batch_size = 32
chunk_size = 50 * batch_size
nb_epoch = 2
samples_per_epoch = 50000
num_res_layers = 16

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0001, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, num_res_layers=num_res_layers, batch_size=batch_size)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='mae')

# Load the discriminator
disc_in_dim = data.dims
toonDisc = ToonDiscriminator3(input_shape=disc_in_dim)

try:
    toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
    toonDisc.compile(optimizer=opt, loss='categorical_crossentropy')
    toonDisc.summary()

except Exception:
    # Pre-train discriminator
    print('Training discriminator...')
    toonDisc.compile(optimizer=opt, loss='categorical_crossentropy')
    make_trainable(toonDisc, True)
    toonDisc.summary()

    # Create test data
    X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size).next()
    Y_pred = toonAE.predict(X_test, batch_size=batch_size)
    X_test = np.concatenate((Y_test, Y_pred))
    y_test = np.zeros((len(X_test), 2))
    y_test[:len(Y_test), 0] = 1
    y_test[len(Y_test):, 1] = 1

    count = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
        # Prepare training data
        Y_pred = toonAE.predict(X_train, batch_size=batch_size)
        X = np.concatenate((Y_train, Y_pred))
        y = np.zeros((len(X), 2))
        y[:len(Y_train), 0] = 1
        y[len(Y_train):, 1] = 1

        # Train discriminator
        toonDisc.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size)

        # Compute Accuracy
        y_hat = toonDisc.predict(X_test)
        acc_test = compute_accuracy(y_hat, y_test)
        y_hat = toonDisc.predict(X)
        acc_train = compute_accuracy(y_hat, y)
        print("Test-Accuracy: %0.02f Train-Accuracy: %0.02f" % (acc_test, acc_train))

        # Check if stop
        count += chunk_size
        if count > samples_per_epoch:
            break
        if not count % 20:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc-Chunk_{}.hdf5'.format(count)))

    toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc.hdf5'))

# Stick them together
make_trainable(toonDisc, False)
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
im_class = toonDisc(im_recon)
toonGAN = Model(im_input, im_class)
toonGAN.compile(optimizer=opt, loss='categorical_crossentropy')

# Store losses
losses = {"d": [], "g": []}

# Training
print('Adversarial training...')
loss_avg_rate = 0.9
loss_target_ratio = 0.25
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    g_loss_avg = 3
    d_loss_avg = 3
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):
        if d_loss_avg > loss_target_ratio * g_loss_avg:
            # Construct data for discriminator training
            Y_pred = toonAE.predict(X_train)
            X = np.concatenate((Y_train, Y_pred))
            y = np.zeros((len(X), 2))
            y[:len(Y_train), 0] = 1
            y[len(Y_train):, 1] = 1
            # Train discriminator
            make_trainable(toonDisc, True)
            d_loss = toonDisc.train_on_batch(X, y)
            losses["d"].append(d_loss)
            d_loss_avg = loss_avg_rate * d_loss_avg + (1 - loss_avg_rate) * d_loss
            del X, Y_pred, y
        else:
            # Train generator
            y = np.zeros((batch_size, 2))
            y[:, 0] = 1
            make_trainable(toonDisc, False)
            g_loss = toonGAN.train_on_batch(X_train, y)
            losses["g"].append(g_loss)
            g_loss_avg = loss_avg_rate * g_loss_avg + (1 - loss_avg_rate) * g_loss

        # Generate montage of test-images
        if not chunk % 100:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc_GAN-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAE_GAN-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
            montage(np.concatenate(
                (decoded_imgs[:12, :, :] * 0.5 + 0.5, X_train[:12] * 0.5 + 0.5, Y_train[:12] * 0.5 + 0.5)),
                    os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))
        chunk += 1

        print('GAN4 Epoch: {}/{} Batch: {} Discriminator-Loss: {} Generator-Loss: {}'.format(epoch, nb_epoch, chunk,
                                                                                             d_loss_avg,
                                                                                             g_loss_avg))
        sys.stdout.flush()
        del X_train, Y_train, y
        gc.collect()

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN.hdf5'))
