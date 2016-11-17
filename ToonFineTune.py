import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import time
import sys
import gc

from ToonNet import Classifier, make_name, gen_data
from datasets import CIFAR10, TinyImagenet
from utils import generator_queue


# Get the data-set object
data = CIFAR10()
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
nb_epoch = 20

use_gan = True
use_gen = True
if use_gen:
    net_load_name = make_name('ToonGen', num_layers=num_layers)
else:
    net_load_name = make_name('ToonDisc', num_layers=num_layers)

if use_gan:
    net_load_name = '{}_gan'.format(net_load_name)

# Load the net
classifier = Classifier(input_shape=data.dims, num_layers=num_layers, num_classes=data.num_classes,
                        net_load_name=net_load_name, use_gen=use_gen, batch_size=batch_size)
classifier.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(classifier.name, data.name)
print('Training network: {}'.format(net_name))



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
                img_train, labels = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        if use_gen:
            X = gen_data(img_train, np.random.normal(size=(len(img_train),)+data.dims[:2]+(1,)))
        else:
            X = np.concatenate((img_train, img_train), axis=3)

        h = classifier.fit(x=X, y=labels, nb_epoch=1, batch_size=batch_size, verbose=0)
        t_loss = h.history['loss'][0]
        print('Loss: {}'.format(t_loss))

        sys.stdout.flush()

    _stop.set()
    del data_gen_queue, threads
    gc.collect()


# Testing
num_chunks = data.num_val // chunk_size

# Create queue for training data
data_gen_queue, _stop, threads = generator_queue(
    datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
    max_q_size=32,
    nb_worker=8)

correct_preds = []
for chunk in range(num_chunks):
    # Get next chunk of training data from queue
    while not _stop.is_set():
        if not data_gen_queue.empty():
            imgs, labels = data_gen_queue.get()
            break
        else:
            time.sleep(0.05)
    if use_gen:
        X = gen_data(imgs, np.random.normal(size=(len(imgs),) + data.dims[:2] + (1,)))
    else:
        X = np.concatenate((imgs, imgs), axis=3)
    pred = classifier.predict(X, batch_size=batch_size)
    y_pred = np.argmax(pred, axis=1)
    correct_preds.append(np.equal(labels, y_pred))

print(np.mean(correct_preds))
