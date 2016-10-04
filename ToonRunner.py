import tensorflow as tf
import numpy as np
import gc
from keras.utils import generic_utils
from keras.preprocessing.image import ImageDataGenerator

from ToonNet import ToonNet, ToonResNet
from datasets.TinyImagenet import TinyImagenet
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 64
nb_epoch = 1

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
net, _ = ToonResNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid', num_res_layers=8)

# Training
for e in range(nb_epoch):
    print("Epoch {} / {}".format(e+1, nb_epoch))
    # Generate batches of around 5000 samples
    for X_train, Y_train in data.generator_train(batch_size):
        net.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                          samples_per_epoch=X_train.shape[0],
                          nb_epoch=2,
                          verbose=2)
        gc.collect()

# decoded_imgs = net.predict(X_test, batch_size=batch_size)
# decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)
#
#
# montage(X_test[:100, :, :], 'ToonResNet-X')
# montage(decoded_imgs[:100, :, :], 'ToonResNet-Out')
# montage(Y_test[:100, :, :], 'ToonResNet-Y')

net.save('ToonNet_tiny-imagenet.h5')