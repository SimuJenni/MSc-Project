import gc
import os
import sys
import time

import keras.backend as K
import numpy as np

from ToonDataGenerator import ImageDataGenerator, generator_queue
from ToonNet import Generator, Discriminator, GANwDisc, disc_data
from constants import MODEL_DIR, IMG_DIR
from datasets import ImagenetToon
from utils import montage

# Training parameters
batch_size = 128
chunk_size = 8 * batch_size
num_chunks = 2000000 // chunk_size
nb_epoch = 1
r_weight = 20.0
e_weight = 1.0
loss_target_ratio = 0.1
num_train = num_chunks * chunk_size
num_res_g = 16
num_res_d = 0
layer = [5]
learning_rate = 0.0001
w_outter = False
p_wise_disc = False
disc_with_x = False
big_fd = True
activation = 'relu'
sigma = K.variable(value=0.2, name='sigma')
noise_lower_factor = 0.95

# Get the data-set object
data = ImagenetToon(num_train=num_train, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, load_weights=True, num_res=num_res_g, w_outter=w_outter, activation=activation)
discriminator = Discriminator(data.dims, load_weights=True, train=True, p_wise_out=p_wise_disc, withx=disc_with_x,
                              noise=sigma, num_res=num_res_d, big_f=big_fd)
gan, gen_gan, disc_gan = GANwDisc(data.dims, load_weights=True, recon_weight=r_weight,
                                  withx=disc_with_x, num_res_g=num_res_g, enc_weight=e_weight,
                                  layers=layer, learning_rate=learning_rate, w_outter=w_outter,
                                  p_wise_out=p_wise_disc, activation=activation, noise=sigma,
                                  num_res_d=num_res_d, big_f=big_fd)
disc_enc = Discriminator(data.dims, load_weights=False, train=False, layers=layer, num_res=num_res_d, big_f=big_fd, noise=sigma)

net_specs = 'rw{}_ew{}_l{}_ltr{}'.format(r_weight, e_weight, layer, loss_target_ratio)
gen_name = '{}_{}'.format(gen_gan.name, net_specs)
disc_name = '{}_{}'.format(disc_gan.name, net_specs)

# Paths for storing the weights
gen_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(gen_name))
disc_weights = os.path.join(MODEL_DIR, '{}.hdf5'.format(disc_name))

# Store losses
losses = {"d": [], "g": []}

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size, target_size=data.target_size).next()
montage(X_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(gen_name)))

# Training
print('Adversarial training: {}'.format(gen_name))
g_loss = None
d_loss = None
dl_thresh = -np.log(0.5)
max_skip_dtrain = 10
lower_noise_count = 2

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
        max_q_size=16,
        nb_worker=4)
    skip_dtrain_count = 0
    dtrain_count = 0

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                X_train, Y_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        update_disc = False

        if not d_loss or skip_dtrain_count > max_skip_dtrain - 1 or d_loss > g_loss * loss_target_ratio or g_loss < dl_thresh:
            print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
            # Update the weights
            generator.set_weights(gen_gan.get_weights())

            # Construct data for discriminator training
            Yd = generator.predict(X_train, batch_size=batch_size)
            Xd_train, yd_train = disc_data(X_train, Y_train, Yd, p_wise=p_wise_disc, with_x=disc_with_x)

            # Train discriminator
            h = discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)
            d_loss = h.history['loss'][0]

            # Record and print loss
            losses["d"].append(d_loss)
            print('d-Loss: {}'.format(d_loss))

            update_disc = True
            dtrain_count += 1
            if dtrain_count > lower_noise_count - 1:
                if skip_dtrain_count < max_skip_dtrain:
                    new_sigma = K.get_value(sigma) * noise_lower_factor
                    print('Lowering noise-level to {}'.format(new_sigma))
                    K.set_value(sigma, new_sigma)
                dtrain_count = 0
            skip_dtrain_count = 0

            del Xd_train, yd_train, Yd
        else:
            skip_dtrain_count += 1
            dtrain_count = 0

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))

        # Update the weights
        if update_disc:
            disc_gan.set_weights(discriminator.get_weights())
            disc_enc.set_weights(discriminator.get_weights())

        # Train generator
        Yg_train = disc_enc.predict(Y_train)
        if p_wise_disc:
            Yg_train[-1] = np.ones((len(Y_train), 4, 4, 1))
        else:
            Yg_train[-1] = np.ones((len(Y_train), 1))
        h = gan.fit(x=X_train, y=Yg_train + [Y_train], nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        g_loss = h.history['{}_loss_{}'.format(gan.output_names[-2], len(layer) + 1)][0]
        e_loss = h.history['{}_loss_{}'.format(gan.output_names[-3], len(layer))][0]
        r_loss = h.history['{}_loss'.format(gan.output_names[-1])][0]

        # Record and print loss
        losses["g"].append(g_loss)
        print('Loss: {} g-Loss: {} r-Loss: {} e-Loss: {}'.format(t_loss, g_loss, r_loss, e_loss))

        # Generate montage of test-images
        if not chunk % 50:
            generator.set_weights(gen_gan.get_weights())
            decoded_imgs = generator.predict(X_test[:batch_size], batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR, '{}-Epoch:{}-Chunk:{}.jpeg'.format(gen_name, epoch, chunk)))
            # Save the weights
            discriminator.save_weights(disc_weights)
            gen_gan.save_weights(gen_weights)

        sys.stdout.flush()
        del X_train, Y_train
        gc.collect()

    # Save the weights
    discriminator.save_weights(disc_weights)
    gen_gan.save_weights(gen_weights)

    _stop.set()
