import os
import sys
import gc

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Discriminator, Generator
from constants import MODEL_DIR
from datasets import Imagenet


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


def disc_data(X, Y, Yd, p_wise=False, with_x=False):
    if with_x:
        Xd = np.concatenate((np.concatenate((X, Y), axis=3), np.concatenate((X, Yd), axis=3)))
    else:
        Xd = np.concatenate((Y, Yd))

    if p_wise:
        yd = np.concatenate((np.ones((len(Y), 4, 4, 1)), np.zeros((len(Y), 4, 4, 1))), axis=0)
    else:
        yd = np.zeros((len(Y) + len(Yd), 1))
        yd[:len(Y)] = 1
    return Xd, yd


batch_size = 64
chunk_size = 32 * batch_size
num_train = 200000
num_res_g = 16
disc_with_x = True
p_wise = True

# Get the data-set object
data = Imagenet(num_train=num_train, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, load_weights=True, w_outter=False, num_res=num_res_g)
discriminator = Discriminator(data.dims, load_weights=False, train=True, withx=disc_with_x, p_wise_out=p_wise)
discriminator.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(discriminator.name, data.name)
print('Training network: {}'.format(net_name))

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size).next()
Y_pred = generator.predict(X_test, batch_size=batch_size)
Xd_test, yd_test = disc_data(X_test, Y_test, Y_pred, p_wise=p_wise, with_x=disc_with_x)

for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size,
                                                    target_size=data.target_size):
    # Prepare training data
    Y_pred = generator.predict(X_train, batch_size=batch_size)
    Xd_train, yd_train = disc_data(X_train, Y_train, Y_pred, p_wise=p_wise, with_x=disc_with_x)

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
