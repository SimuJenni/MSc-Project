import gc
import os
import sys
import time

import keras.backend as K
import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import GANAE, gen_data
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon
from utils import montage, generator_queue

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.75, 1.0],
    fill_mode='nearest',
    horizontal_flip=True
)

# Training parameters
num_layers = 3
batch_size = 200
chunk_size = 4 * batch_size
num_chunks = data.num_train // chunk_size
nb_epoch = 30
load_weights = False
noise = K.variable(value=0.5, name='sigma')
noise_decay_rate = 0.95
merge_order = K.variable(value=np.int16(0), name='merge_order', dtype='int16')

# Load the models
gGAN = GANAE(data.dims, order=merge_order, batch_size=batch_size, num_layers=num_layers, train_disc=False)
dGAN = GANAE(data.dims, order=merge_order, batch_size=batch_size, num_layers=num_layers, train_disc=True)

# Create test data
toon_test, edge_test, im_test = datagen.flow_from_directory(data.val_dir, batch_size=chunk_size,
                                                            target_size=data.target_size).next()
X_test = [gen_data(toon_test, edge_test), im_test]
montage(toon_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-toon.jpeg'.format(gGAN.name, data.name)))
montage(np.squeeze(edge_test[:100]), os.path.join(IMG_DIR, '{}-{}-edge.jpeg'.format(gGAN.name, data.name)), gray=True)
montage(im_test[:100] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-{}-images.jpeg'.format(gGAN.name, data.name)))

# Training
print('GAN training: {}'.format(gGAN.name))

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    train_disc = True
    count_skip = 0

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
        max_q_size=32,
        nb_worker=8)

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                toon_train, edge_train, img_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        X = [gen_data(toon_train, edge_train), img_train]

        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))
        dGAN.set_weights(gGAN.get_weights())

        # Train generator
        if chunk % 2 == 0:
            y = [np.zeros((len(toon_train), 1)), img_train]
        else:
            y = [np.ones((len(toon_train), 1)), img_train]

        h = dGAN.fit(x=X, y=y, nb_epoch=1, batch_size=batch_size, verbose=0)
        print(h.history)

        print('Epoch {}/{} Chunk {}: Training Generator...'.format(epoch, nb_epoch, chunk))
        gGAN.set_weights(dGAN.get_weights())

        # Train generator
        if chunk % 2 == 0:
            y = [np.ones((len(toon_train), 1)), img_train]
        else:
            y = [np.zeros((len(toon_train), 1)), img_train]

        h = gGAN.fit(x=X, y=y, nb_epoch=1, batch_size=batch_size, verbose=0)
        print(h.history)

        # Generate montage of test-images
        if not chunk % 25:
            _, decoded_imgs = gGAN.predict(X_test, batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR,
                                 '{}-{}-Epoch:{}-Chunk:{}-big_m_bigF.jpeg'.format(gGAN.name, data.name, epoch, chunk)))
            _, decoded_imgs = dGAN.predict(X_test, batch_size=batch_size)
            montage(decoded_imgs[:100] * 0.5 + 0.5,
                    os.path.join(IMG_DIR,
                                 '{}-{}-Epoch:{}-Chunk:{}-big_m_bigF.jpeg'.format(dGAN.name, data.name, epoch, chunk)))
            del toon_train, img_train, h, decoded_imgs
            gc.collect()
        sys.stdout.flush()
        K.set_value(merge_order, np.int16(chunk % 2))

    _stop.set()
    del data_gen_queue, threads
    gc.collect()
