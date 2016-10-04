import gc

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from ToonNet import ToonNet
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 24
nb_epoch = 1

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
toon_net, encoder = ToonNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid',
                            num_res_layers=10)
toon_net.summary()  # For debugging

# Define objective and solver
toon_net.compile(optimizer='adam', loss='mse')

# Training
for e in range(nb_epoch):
    print("Epoch {} / {}".format(e + 1, nb_epoch))
    # Generate batches of around 5000 samples
    for X_train, Y_train, X_test, Y_test in data.generator(batch_size):
        toon_net.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                               samples_per_epoch=X_train.shape[0],
                               nb_epoch=1,
                               validation_data=(X_test, Y_test))
        gc.collect()

# Test
decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)

montage(X_test[:100, :, :], 'ToonResNet-X')
montage(decoded_imgs[:100, :, :], 'ToonResNet-Out')
montage(Y_test[:100, :, :], 'ToonResNet-Y')

toon_net.save('ToonNet_tiny-imagenet.h5')
