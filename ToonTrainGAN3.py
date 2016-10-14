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
        print(l)
        #TODO check if works with res layers too
        l.trainable = val


def compute_accuracy(y_hat, y):
    y_hat = y_hat > 0.5
    acc = np.mean(y_hat == y)
    return acc


batch_size = 32
chunk_size = 50*batch_size
nb_epoch = 2
num_res_layers = 16

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, num_res_layers=num_res_layers, batch_size=batch_size)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='binary_crossentropy')

# Load the discriminator
toonDisc = ToonDiscriminator2(input_shape=data.dims)
#toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')
toonDisc.summary()

# Stick them together
make_trainable(toonDisc, False)
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
im_class = toonDisc(im_recon)
toonGAN = Model(im_input, im_class)
toonGAN.compile(optimizer=opt, loss='binary_crossentropy')

# Pre-train discriminator
make_trainable(toonDisc, True)
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size).next()
Y_pred = toonAE.predict(X_test, batch_size=batch_size)
X_test = np.concatenate((Y_test, Y_pred))
y_test = np.array([1]*len(Y_test) + [0]*len(Y_pred))

for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
    Y_pred = toonAE.predict(X_train, batch_size=batch_size)
    X = np.concatenate((Y_train, Y_pred))
    y = np.array([1]*len(Y_train) + [0]*len(Y_pred))

    toonDisc.fit(X, y, nb_epoch=2, batch_size=batch_size)

    # Compute Accuracy
    y_hat = toonDisc.predict(X_test)
    acc_test = compute_accuracy(y_hat, y_test)
    y_hat = toonDisc.predict(X_train)
    acc_train = compute_accuracy(y_hat, y)
    print("Test-Accuracy: %0.02f Train-Accuracy: %0.02f" % (acc_test, acc_train))

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
