import gc
import os

import numpy as np
import tensorflow as tf
from keras.backend import switch
from keras.layers import Input, merge, Lambda
from keras.models import Model
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
chunk_size = 200 * batch_size
nb_epoch = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size)
# toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='mae')

# Load the discriminator
disc_in_dim = (data.dims[0], data.dims[1], 6)
toonDisc = ToonDiscriminator(input_shape=disc_in_dim)
# toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')

# Stick them together
X_input = Input(shape=data.dims)
Y_input = Input(shape=data.dims)
order_input = Input(batch_shape=(1,), dtype='int32')
im_recon = toonAE(X_input)


def concat_order(x):
    return switch(tf.reshape(order_input, []), merge([x, Y_input], mode='concat'),
                  merge([Y_input, x], mode='concat'))


disc_in = Lambda(concat_order)(im_recon)
im_class = toonDisc(disc_in)
toonGAN = Model([X_input, Y_input, order_input], im_class)
toonGAN.compile(optimizer=opt, loss='binary_crossentropy')

# Training
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):

        # Construct data for training
        y_disc = [1] * len(X_train) + [0] * len(Y_train)
        y_gen = [0] * len(X_train) + [1] * len(Y_train)
        train_input = [np.repeat(X_train, 2, axis=0), np.repeat(Y_train, 2, axis=0), y_disc]

        # Train discriminator
        toonDisc.trainable = True
        toonAE.trainable = False
        toonGAN.fit(train_input, y_disc, batch_size=batch_size, nb_epoch=1)

        # Train generator
        toonDisc.trainable = False
        toonAE.trainable = True
        toonGAN.fit(train_input, y_gen, batch_size=batch_size, nb_epoch=1)

        # Generate montage of test-images
        chunk += 1
        if not chunk % 1:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonGANDisc-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonGANAE-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
            montage(decoded_imgs[:(2 * batch_size), :, :] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))

        del X_train, Y_train, train_input, y_gen, y_disc
        gc.collect()

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN.hdf5'))
