import os
import sys

import numpy as np

from DataGenerator import ImageDataGenerator
from ToonNet import Discriminator, Generator
from constants import MODEL_DIR
from datasets import Imagenet


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


batch_size = 64
chunk_size = 50 * batch_size
num_train = 200000

# Get the data-set object
data = Imagenet(num_train=num_train, target_size=(128, 128))
datagen = ImageDataGenerator()

# Load the models
generator = Generator(data.dims, load_weights=True, w_outter=False)
discriminator = Discriminator(data.dims, load_weights=False, train=True)
discriminator.summary()

# Name used for saving of model and outputs
net_name = '{}-{}'.format(discriminator.name, data.name)
print('Training network: {}'.format(net_name))

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size).next()
Y_pred = generator.predict(X_test, batch_size=batch_size)
X_test = np.concatenate((Y_test, Y_pred))
y_test = np.zeros((len(Y_test) + len(Y_pred), 1))
y_test[:len(Y_test)] = 1

training_done = False
while not training_done:
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size,
                                                        target_size=data.target_size):
        # Prepare training data
        Y_pred = generator.predict(X_train, batch_size=batch_size)
        X = np.concatenate((Y_train, Y_pred))
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

discriminator.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))
