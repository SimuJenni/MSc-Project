import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import DiscriminatorWithX, Generator
from constants import MODEL_DIR
from datasets import Imagenet


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


batch_size = 32
chunk_size = 50 * batch_size
f_dims = [64, 128, 256, 512, 1024]

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, batch_size, load_weights=True, f_dims=f_dims)
discriminator = DiscriminatorWithX(data.dims, load_weights=False, f_dims=f_dims)

# Pre-train discriminator
print('Training discriminator...')

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size).next()
Y_pred = generator.predict(X_test, batch_size=batch_size)
X_test = np.concatenate((np.concatenate((X_test, Y_test), axis=3), np.concatenate((X_test, Y_pred), axis=3)))
y_test = np.zeros((len(Y_test) + len(Y_pred), 1))
y_test[:len(Y_test)] = 1

training_done = False
while not training_done:
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
        # Prepare training data
        Y_pred = generator.predict(X_train, batch_size=batch_size)
        X = np.concatenate((np.concatenate((X_train, Y_test), axis=3), np.concatenate((X_train, Y_pred), axis=3)))
        y = np.zeros((len(Y_train) + len(Y_pred), 1))
        y[:len(Y_train)] = 1

        # Train discriminator
        discriminator.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=0)
        train_loss = discriminator.evaluate(X, y, batch_size=batch_size, verbose=0)
        test_loss = discriminator.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print('Test-Loss: %0.02f Train-Loss: %0.02f' % (test_loss, train_loss))

        # Compute Accuracy
        y_hat = discriminator.predict(X_test)
        acc_test = compute_accuracy(y_hat, y_test)
        y_hat = discriminator.predict(X)
        acc_train = compute_accuracy(y_hat, y)
        print("Test-Accuracy: %0.02f Train-Accuracy: %0.02f" % (acc_test, acc_train))
        sys.stdout.flush()

        # Check if training can be stopped
        if test_loss == 0.0 or acc_test == 1.0:
            training_done = True
            break

discriminator.save_weights(os.path.join(MODEL_DIR, 'ToonDiscWithX.hdf5'))
