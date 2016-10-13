import gc
import os
import sys

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend import switch
from keras.layers import Input, merge, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

sess = tf.Session()
K.set_session(sess)

batch_size = 128
nb_epoch = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# do not initialize variables on the fly
K.manual_variable_initialization(True)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size, num_res_layers=16)
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')

# Load the discriminator
disc_in_dim = (data.dims[0], data.dims[1], 6)
toonDisc = ToonDiscriminator(input_shape=disc_in_dim)

# Stick them together
X_input = Input(shape=data.dims, name='X_train')
Y_input = tf.placeholder(tf.float32, shape=(None,) + data.dims, name='Y_train')
order_label = tf.placeholder(tf.int32, shape=())
im_recon = toonAE(X_input)


def concat_order(x):
    return switch(tf.reshape(order_label, []), merge([x, Y_input], mode='concat'),
                  merge([Y_input, x], mode='concat'))


disc_in = Lambda(concat_order)(im_recon)
pred = toonDisc(disc_in)
toonGAN = Model(X_input, pred)

# placeholder for training targets
targets = tf.placeholder(tf.float32, shape=(None, 1), name='targets')

# loss objective
loss = tf.reduce_mean(binary_crossentropy(targets, pred))

# apply regularizers if any
if toonGAN.regularizers:
    total_loss = loss * 1.  # copy tensor
    for regularizer in toonGAN.regularizers:
        total_loss = regularizer(total_loss)
else:
    total_loss = loss

# set up TF optimizer
optimizer = tf.train.AdamOptimizer(0.001)

# Batchnorm updates
with tf.control_dependencies(toonGAN.updates):
    total_loss = tf.identity(total_loss)

# Compute gradients with respect to the loss.
grads = optimizer.compute_gradients(total_loss)

apply_gradients_op = optimizer.apply_gradients(grads)

with tf.control_dependencies([apply_gradients_op]):
    train_op = tf.identity(total_loss, name='train_op')

init_op = tf.initialize_all_variables()

with sess.as_default():
    K.set_session(sess)
    sess.run(init_op)
    # Training
    for epoch in range(nb_epoch):
        chunk = 0
        for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):

            # Train discriminator
            toonDisc.trainable = True
            toonAE.trainable = False

            feed_dict = {X_input: X_train, Y_input: Y_train, order_label: 1, targets: np.ones([batch_size, 1])}
            _, l_disc1 = sess.run([train_op, total_loss], feed_dict=feed_dict)
            feed_dict = {X_input: X_train, Y_input: Y_train, order_label: 0, targets: np.zeros([batch_size, 1])}
            _, l_disc2 = sess.run([train_op, total_loss], feed_dict=feed_dict)

            # Train generator
            toonDisc.trainable = False
            toonAE.trainable = True

            feed_dict = {X_input: X_train, Y_input: Y_train, order_label: 1, targets: np.zeros([batch_size, 1])}
            _, l_gen1 = sess.run([train_op, total_loss], feed_dict=feed_dict)
            feed_dict = {X_input: X_train, Y_input: Y_train, order_label: 0, targets: np.ones([batch_size, 1])}
            _, l_gen2 = sess.run([train_op, total_loss], feed_dict=feed_dict)

            # Generate montage of test-images
            if not chunk % 50:
                toonDisc.save_weights(
                    os.path.join(MODEL_DIR, 'ToonGAN2Disc-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
                toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonGAN2AE-Epoch:{}-Chunk:{}.hdf5'.format(epoch, chunk)))
                decoded_imgs = toonAE.predict(X_train[:(2 * batch_size)], batch_size=batch_size)
                montage(decoded_imgs[:(2 * batch_size), :, :] * 0.5 + 0.5,
                        os.path.join(IMG_DIR, 'GAN2-Epoch:{}-Chunk:{}.jpeg'.format(epoch, chunk)))

            chunk += 1
            print('Epoch: {}/{} Batch: {} Discriminator-Loss: {} Generator-Loss: {}'.format(epoch, nb_epoch, chunk,
                                                                                            l_disc1 + l_disc2,
                                                                                            l_gen1 + l_gen2))
            sys.stdout.flush()

            del X_train, Y_train
            gc.collect()

    toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDiscGAN2.hdf5'))
    toonAE.save_weights(os.path.join(MODEL_DIR, 'ToonAEGAN2.hdf5'))
