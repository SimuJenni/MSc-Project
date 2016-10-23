import os
import sys

from DataGenerator import ImageDataGenerator
from ToonNet import Encoder
from constants import MODEL_DIR
from datasets import Imagenet

batch_size = 64
chunk_size = 50 * batch_size
num_epochs = 2

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the models
encoder, generator = Encoder(data.dims, load_weights=False, train=True)
print('Training encoder...')

# Get test data
X_test, _ = datagen.flow_from_directory(data.train_dir, batch_size=chunk_size).next()

training_done = False
for ep in range(num_epochs):
    for X_train, _ in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):
        # Train Encoder
        generator.fit(X_train, X_train, nb_epoch=1, batch_size=batch_size, verbose=0)
        train_loss = generator.evaluate(X_train, X_train, batch_size=batch_size, verbose=0)
        test_loss = generator.evaluate(X_test, X_test, batch_size=batch_size, verbose=0)
        print('Test-Loss: %0.03f Train-Loss: %0.03f' % (test_loss, train_loss))
        sys.stdout.flush()

    encoder.save_weights(os.path.join(MODEL_DIR, '{}_Epoch{}.hdf5'.format(encoder.name, ep)))
    generator.save_weights(os.path.join(MODEL_DIR, '{}_Epoch{}.hdf5'.format(generator.name, ep)))
