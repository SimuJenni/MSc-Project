import os
import numpy as np
import gc

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
chunk_size = 200*batch_size
nb_epoch = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator(horizontal_flip=True)

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer=opt, loss='binary_crossentropy')

# Load the discriminator
toonDisc = ToonDiscriminator(input_shape=data.dims)
toonDisc.load_weights('/home/sj09l405/MSc-Project/ToonDisc.hdf5')
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')

# Stick them together
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
im_class = toonDisc(im_recon)
toonGAN = Model(im_input, im_class)
toonGAN.compile(optimizer=opt, loss='binary_crossentropy')

# Training
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
        Y_pred = toonAE.predict(X_train, batch_size=batch_size)

        # Construct data for discriminator training
        X_disc = np.concatenate((Y_train, Y_pred))
        y = [1] * len(Y_train) + [0] * len(Y_pred)

        # Train discriminator
        toonDisc.trainable = True
        toonDisc.fit(X_disc, y, batch_size=batch_size, nb_epoch=1)

        # Train generator
        toonDisc.trainable = False
        toonGAN.fit(X_train, [1] * len(X_train), batch_size=batch_size, nb_epoch=1)

        # Generate montage of test-images
        chunk += 1
        if not chunk % 10:
            toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAE-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
            decoded_imgs = toonAE.predict(X_train[:49], batch_size=batch_size)
            montage(decoded_imgs[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, 'GAN-Epoch:{}-Chunk:{}'.format(epoch, chunk)))

        del X_train, Y_train, X_disc, Y_pred, y
        gc.collect()

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN.hdf5'))
toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN.hdf5'))
