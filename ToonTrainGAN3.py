import gc
import os

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


batch_size = 32
nb_epoch = 2
num_res_layers = 16

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt_g = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, num_res_layers=num_res_layers, batch_size=batch_size)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='binary_crossentropy')

# Load the discriminator
toonDisc = ToonDiscriminator(input_shape=data.dims)
#toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
toonDisc.compile(optimizer=opt, loss='categorical_crossentropy')

# Stick them together
make_trainable(toonDisc, False)
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
im_class = toonDisc(im_recon)
toonGAN = Model(im_input, im_class)
toonGAN.compile(optimizer=opt, loss='categorical_crossentropy')

# Pre-train discriminator
for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=50*batch_size):
    Y_pred = toonAE.predict(X_train, batch_size=batch_size)
    X = np.concatenate((Y_train, Y_pred))
    y = np.zeros([len(X), 2])
    y[:len(Y_train), 1] = 1
    y[len(Y_train):, 0] = 1

    make_trainable(toonDisc, True)
    toonDisc.fit(X, y, nb_epoch=1, batch_size=batch_size)

    # Compute Accuracy
    y_hat = toonDisc.predict(X)
    print(y_hat)
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot
    print("Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc.hdf5'))

# Store losses
losses = {"d": [], "g": []}

# Training
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):
        Y_pred = toonAE.predict(X_train)

        # Construct data for discriminator training
        X = np.concatenate((Y_train, Y_pred))
        y = np.zeros([2 * batch_size, 2])
        y[0:batch_size, 1] = 1
        y[batch_size:, 0] = 1

        # Train discriminator
        make_trainable(toonDisc, True)
        d_loss = toonDisc.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # Train generator
        y = np.zeros([batch_size, 2])
        y[:, 1] = 1
        make_trainable(toonDisc, False)
        g_loss = toonGAN.train_on_batch(X_train, y)

        # Generate montage of test-images
        if not chunk % 5:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc_GAN-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAE_GAN-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
            montage(decoded_imgs[:(2 * batch_size), :, :] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))

        chunk += 1

        print('Epoch: {}/{} Batch: {} Discriminator-Loss: {} Generator-Loss: {}'.format(epoch, nb_epoch, chunk,
                                                                                        d_loss,
                                                                                        g_loss))

        del X_train, Y_train, X, Y_pred, y
        gc.collect()

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN.hdf5'))
