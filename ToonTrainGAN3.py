import gc
import os

import numpy as np
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator, ToonDiscriminator2
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        #TODO check if works with res layers too
        l.trainable = val


batch_size = 32
nb_epoch = 2
num_res_layers = 16

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, num_res_layers=num_res_layers, batch_size=batch_size)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='binary_crossentropy')

# Load the discriminator
toonDisc = ToonDiscriminator2(input_shape=(data.dims[0], data.dims[1], 6))
#toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
toonDisc.compile(optimizer=opt, loss='categorical_crossentropy')
toonDisc.summary()

# Stick them together
make_trainable(toonDisc, False)
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
disc_in = merge([im_recon, im_input], mode='concat')
im_class = toonDisc(disc_in)
toonGAN = Model(im_input, im_class)
toonGAN.compile(optimizer=opt, loss='categorical_crossentropy')

# Pre-train discriminator
make_trainable(toonDisc, True)
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=50*batch_size).next()
Y_pred = toonAE.predict(X_test, batch_size=batch_size)
X_test = np.concatenate((np.concatenate((Y_test, X_test), axis=3),
                              np.concatenate((Y_pred, X_test), axis=3)))
y_test = np.zeros([len(X_test), 2])
y_test[:len(Y_test), 1] = 1
y_test[len(Y_test):, 0] = 1

for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=50*batch_size):
    Y_pred = toonAE.predict(X_train, batch_size=batch_size)
    X_train = np.concatenate((np.concatenate((Y_train, X_train), axis=3),
                              np.concatenate((Y_pred, X_train), axis=3)))
    y = np.zeros([len(X_train), 2])
    y[:len(Y_train), 1] = 1
    y[len(Y_train):, 0] = 1

    toonDisc.fit(X_train, y, nb_epoch=1, batch_size=batch_size)

    # Compute Accuracy
    y_hat = toonDisc.predict(X_test)
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y_test, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y_test.shape[0]
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
