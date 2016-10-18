import gc
import os
import sys

import numpy as np
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.objectives import binary_crossentropy

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE2, ToonDiscriminator2
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

batch_size = 32
chunk_size = 50 * batch_size
nb_epoch = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Placeholder for input
x = tf.placeholder(tf.float32, shape=(batch_size,) + data.dims)

# placeholder for training targets
targets = tf.placeholder(tf.float32, shape=(None, 1), name='targets')

# Load the auto-encoder
toonAE = ToonAE2(input_shape=data.dims, batch_size=batch_size, bn_mode=2)
toonAE.load_weights(os.path.join(MODEL_DIR, 'ToonAE2.hdf5'))

# Load the discriminator
disc_in_dim = data.dims[:2] + (6,)
toonDisc = ToonDiscriminator2(input_shape=disc_in_dim)
toonDisc.load_weights(os.path.join(MODEL_DIR, 'ToonDisc_m3_converge.hdf5'))

# Stick them together
make_trainable(toonDisc, False)
im_input = Input(shape=data.dims)
im_recon = toonAE(im_input)
disc_in = merge([im_input, im_recon], mode='concat')
preds = toonDisc(disc_in)
toonGAN = Model(input=im_input, output=preds)

update_ops_GAN = []
for old_value, new_value in toonGAN.updates:
    update_ops_GAN.append(tf.assign(old_value, new_value))

update_ops_Disc = []
for old_value, new_value in toonDisc.updates:
    update_ops_Disc.append(tf.assign(old_value, new_value))


disc_loss = tf.reduce_mean(binary_crossentropy(targets, preds))
disc_loss = control_flow_ops.with_dependencies(update_ops_Disc, disc_loss)

gan_loss = tf.reduce_mean(binary_crossentropy(targets, preds))
disc_loss = control_flow_ops.with_dependencies(update_ops_Disc, disc_loss)

optimizer = tf.train.AdamOptimizer(0.0002)

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
            X = np.concatenate((np.concatenate((X_train, Y_train), axis=3),
                                (np.concatenate((X_train, Y_pred), axis=3))))
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
