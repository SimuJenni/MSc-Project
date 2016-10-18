import gc
import os
import sys

import numpy as np
from keras.layers import Input, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import switch, reshape

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonDiscriminator2
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from utils import montage


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


batch_size = 32
chunk_size = 10 * batch_size
nb_epoch = 2
num_res_layers = 8

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, num_res_layers=num_res_layers, batch_size=batch_size)

# Load the discriminator
disc_in_dim = data.dims[:2] + (6,)
toonDisc = ToonDiscriminator2(input_shape=disc_in_dim)

# Stick them together
im_input = Input(shape=data.dims)
gt_input = Input(shape=data.dims)
order_input = Input(shape=(1,))

im_recon = toonAE(im_input)

d_in1 = switch(reshape(order_input, []), im_recon, gt_input)
d_in2 = switch(reshape(order_input, []), gt_input, im_recon)

disc_in = merge([d_in1, d_in2], mode='concat')
im_class = toonDisc(disc_in)

toonGAN = Model(input=[im_input, gt_input, order_input], output=[im_class, im_recon])
theta = 0.1
toonGAN.compile(optimizer=opt, loss=['binary_crossentropy', 'mse'], loss_weights=[1.0, theta])

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
        # Train discriminator
        make_trainable(toonDisc, True)
        make_trainable(toonAE, False)
        y = np.random.choice([0, 1], size=(len(X_train), 1))
        d_loss, _ = toonGAN.train_on_batch([X_train, Y_train, y], [y, Y_train])
        # Train generator
        make_trainable(toonDisc, False)
        make_trainable(toonAE, True)
        g_loss, r_loss = toonGAN.train_on_batch([X_train, Y_train, y], [1-y, Y_train])

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
