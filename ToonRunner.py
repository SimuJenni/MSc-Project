import tensorflow as tf
import numpy as np
import gc

from ToonNet import ToonNet
from datasets.TinyImagenet import TinyImagenet
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 250
nb_epoch = 20


# Get the data-set object
data = TinyImagenet()

# Load the net
net, _ = ToonNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid')

# Min reconstruction
for e in range(nb_epoch):
    print("Epoch {} / {}".format(e+1, nb_epoch))
    for X_train, Y_train, X_test, Y_test in data.generator(batch_size):
        num_train = X_train.shape[0]
        for i in range(num_train//batch_size):
            net.train_on_batch(X_train[i*batch_size:(i+1)*batch_size],
                               Y_train[i * batch_size:(i + 1) * batch_size])
        #net.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_test, Y_test))
        gc.collect()

decoded_imgs = net.predict(X_test, batch_size=batch_size)
decoded_imgs = np.clip(decoded_imgs, 0.0, 1.0)



montage(X_test[:100, :, :], 'ToonResNet-X')
montage(decoded_imgs[:100, :, :], 'ToonResNet-Out')
montage(Y_test[:100, :, :], 'ToonResNet-Y')

net.save('ToonNet_tiny-imagenet.h5')