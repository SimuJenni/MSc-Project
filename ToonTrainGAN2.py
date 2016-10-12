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
from datasets.TinyImagenet import TinyImagenet
from utils import montage

batch_size = 32
nb_epoch = 2

# Get the data-set object
data = TinyImagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

#TODO: Shit doesn't work with keras... Try with tensorflow placeholders!

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
X_input = Input(shape=data.dims, name='X_train')
Y_input = Input(shape=data.dims, name='Y_train')
order_input = Input(batch_shape=(1,1), dtype='int32', name='Order_Label')
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
    chunk = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size, target_size=(64, 64)):
        one = np.asarray([1]*len(X_train)).reshape((len(X_train), 1))
        zero = np.asarray([0]*len(X_train)).reshape((len(X_train), 1))

        # Train discriminator
        toonDisc.trainable = True
        toonAE.trainable = False
        l_disc1 = toonGAN.train_on_batch({'X_train': X_train, 'Y_train': Y_train, 'Order_Label': one}, one)
        l_disc2 = toonGAN.train_on_batch({'X_train': X_train, 'Y_train': Y_train, 'Order_Label': zero}, zero)

        # Train generator
        toonDisc.trainable = False
        toonAE.trainable = True
        l_gen1 = toonGAN.train_on_batch({'X_train': X_train, 'Y_train': Y_train, 'Order_Label': one}, zero)
        l_gen2 = toonGAN.train_on_batch({'X_train': X_train, 'Y_train': Y_train, 'Order_Label': zero}, one)

        # Generate montage of test-images
        chunk += 1
        if not chunk % 1000:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonGANDisc-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonGANAE-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
            montage(decoded_imgs[:(2 * batch_size), :, :] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))

        print('Epoch: {}/{} Batch: {} Discriminator-Loss: {} Generator-Loss: {}'.format(epoch, nb_epoch, chunk, l_disc1+l_disc2, l_gen1+l_gen2))

        del X_train, Y_train
        gc.collect()

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN.hdf5'))
