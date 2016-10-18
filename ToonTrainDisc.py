import os
import sys

import numpy as np
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonDiscriminator2, ToonAE2
from constants import MODEL_DIR
from datasets.Imagenet import Imagenet


def compute_accuracy(y_hat, y):
    return 1.0 - np.mean(np.abs(np.round(y_hat) - y))


batch_size = 32
chunk_size = 50 * batch_size
nb_epoch = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Define optimizer
opt = Adam(lr=0.0002, beta_1=0.5)

# Load the auto-encoder
toonAE = ToonAE2(input_shape=data.dims, batch_size=batch_size, bn_mode=2)
toonAE.load_weights(os.path.join(MODEL_DIR, 'ToonAE2.hdf5'))

# Load the discriminator
disc_in_dim = data.dims[:2] + (6,)
toonDisc = ToonDiscriminator2(input_shape=disc_in_dim)

# Pre-train discriminator
print('Training discriminator...')
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')
toonDisc.summary()

# Create test data
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size).next()
Y_pred = toonAE.predict(X_test, batch_size=batch_size)
X_test = np.concatenate((np.concatenate((X_test, Y_test), axis=3),
                         (np.concatenate((X_test, Y_pred), axis=3))))
y_test = np.zeros((len(Y_test) + len(Y_pred), 1))
y_test[:len(Y_test)] = 1

training_done = False
while not training_done:
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
        # Prepare training data
        Y_pred = toonAE.predict(X_train, batch_size=batch_size)
        X = np.concatenate((np.concatenate((X_train, Y_train), axis=3),
                            (np.concatenate((X_train, Y_pred), axis=3))))
        y = np.zeros((len(Y_train) + len(Y_pred), 1))
        y[:len(Y_train)] = 1

        # Train discriminator
        toonDisc.fit(X, y, nb_epoch=1, batch_size=batch_size, verbose=0)
        train_loss = toonDisc.evaluate(X, y, batch_size=batch_size, verbose=0)
        test_loss = toonDisc.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        print('Test-Loss: %0.02f Train-Loss: %0.02f' % (test_loss, train_loss))

        # Compute Accuracy
        y_hat = toonDisc.predict(X_test)
        acc_test = compute_accuracy(y_hat, y_test)
        y_hat = toonDisc.predict(X)
        acc_train = compute_accuracy(y_hat, y)
        print("Test-Accuracy: %0.02f Train-Accuracy: %0.02f" % (acc_test, acc_train))
        sys.stdout.flush()

        # Check if training can be stopped
        if test_loss == 0.0 or acc_test == 1.0:
            training_done = True
            break

toonDisc.save_weights(os.path.join(MODEL_DIR, 'ToonDisc2.hdf5'))
