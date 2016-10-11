import os
import numpy as np

from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonDiscriminator
from constants import MODEL_DIR
from datasets.Imagenet import Imagenet

batch_size = 32
nb_epoch = 1
chunk_size = 300*batch_size
l_rate = 0.001

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator(horizontal_flip=True)

# Load the net
toonDisc = ToonDiscriminator(input_shape=data.dims)
toonDisc.summary()

# Name used for saving of model and outputs
net_name = '{}-Data:{}-LRate:{}'.format(toonDisc.name, data.name, l_rate)
print('Training network: {}'.format(net_name))

# Define objective and solver
opt = Adam(lr=l_rate)
toonDisc.compile(optimizer=opt, loss='binary_crossentropy')

# Training
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    chunk = 0
    for X_train, Y_train in datagen.flow_from_directory(data.train_dir, batch_size=chunk_size):

        # Construct data for discriminator training
        X_disc = np.concatenate((Y_train, X_train))
        y = [1] * len(Y_train) + [0] * len(X_train)

        # Train discriminator
        toonDisc.fit(X_disc, y, batch_size=batch_size, nb_epoch=1)

        # Generate montage of test-images
        chunk += 1
        if not chunk % 20:
            toonDisc.save_weights(os.path.join(MODEL_DIR, '{}-Epoch:{}-Chunk:{}.hdf5'.format(net_name, epoch, chunk)))

# Save the model
toonDisc.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))
