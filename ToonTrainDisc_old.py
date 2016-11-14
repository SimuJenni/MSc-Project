import gc
import os
import sys

import numpy as np

from ToonDataGenerator import ImageDataGenerator
from ToonNet import Disc, Gen, disc_data
from constants import MODEL_DIR
from datasets import CIFAR10_Toon


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))

# Get the data-set object
data = CIFAR10_Toon()
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=[0.9, 1.0],
    fill_mode='nearest',
    horizontal_flip=True
)

# Training parameters
num_layers = 3
batch_size = 100
chunk_size = 4 * batch_size
num_chunks = data.num_train // chunk_size
nb_epoch = 100

# Load the models
generator = Gen(data.dims, load_weights=False, num_layers=3)
discriminator = Disc(data.dims, load_weights=False)
discriminator.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(discriminator.name, data.name)
print('Training network: {}'.format(net_name))

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size).next()
Y_pred = generator.predict(X_test, batch_size=batch_size)
Xd_test, yd_test = disc_data(X_test, Y_test, Y_pred, p_wise=False, with_x=False)

for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size,
                                                    target_size=data.target_size):
    # Prepare training data
    Y_pred = generator.predict(X_train, batch_size=batch_size)
    Xd_train, yd_train = disc_data(X_train, Y_train, Y_pred, p_wise=False, with_x=False)

    # Train discriminator
    discriminator.fit(Xd_train, yd_train, nb_epoch=1, batch_size=batch_size, verbose=0)
    train_loss = discriminator.evaluate(Xd_train, yd_train, batch_size=batch_size, verbose=0)
    test_loss = discriminator.evaluate(Xd_test, yd_test, batch_size=batch_size, verbose=0)
    print('Test-Loss: %0.02f Train-Loss: %0.02f' % (test_loss, train_loss))

    # Compute Accuracy
    y_hat = discriminator.predict(Xd_test)
    acc_test = compute_accuracy(y_hat, yd_test)
    y_hat = discriminator.predict(Xd_train)
    acc_train = compute_accuracy(y_hat, yd_train)
    print("Test-Accuracy: %0.02f Train-Accuracy: %0.02f" % (acc_test, acc_train))
    sys.stdout.flush()

    # Check if training can be stopped
    if test_loss == 0.0 or acc_test == 1.0:
        training_done = True
        break

    del Y_pred, X_train, Y_train, Xd_train, yd_train
    gc.collect()

discriminator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))
