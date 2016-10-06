import gc

from keras.preprocessing.image import ImageDataGenerator

from ToonNet import ToonNet
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
nb_epoch = 1
nb_epoch_inner = 2
max_train_chunks = 1000  # Chunks of size ~5000
plot_while_train = True
net_name = 'ToonNet-Inits-SumMerge'

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
# toon_net, encoder, decoder = ToonNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid',
#                                      num_res_layers=20)
toon_net, encoder, decoder = ToonNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid',
                                     num_res_layers=10, merge_mode='sum')
toon_net.summary()  # For debugging

# Define objective and solver
toon_net.compile(optimizer='adam', loss='mae')

# Training
chunk_count = 0
for e in range(nb_epoch):
    print("Epoch {} / {}".format(e + 1, nb_epoch))
    # Generate batches of around 5000 samples
    for X_train, Y_train, X_test, Y_test in data.generator(batch_size):
        toon_net.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                               samples_per_epoch=X_train.shape[0],
                               nb_epoch=nb_epoch_inner,
                               validation_data=(X_test, Y_test))
        chunk_count += 1
        if chunk_count > max_train_chunks:
            break

        if plot_while_train:
            # Test images
            decoded_imgs = toon_net.predict(X_test[:batch_size], batch_size=batch_size)
            montage(decoded_imgs[:16, :, :], 'Train: {}-{}'.format(net_name, chunk_count))
        del X_train, Y_train, X_test, Y_test
        gc.collect()

decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
montage(X_test[:100, :, :], '{}-X'.format(net_name))
montage(decoded_imgs[:100, :, :], '{}-Out'.format(net_name))
montage(Y_test[:100, :, :], '{}-Y'.format(net_name))
toon_net.save('{}.h5'.format(net_name))
