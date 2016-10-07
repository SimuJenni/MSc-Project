import gc
import os

from keras.preprocessing.image import ImageDataGenerator

from ToonNet import ToonNet, ToonNetMore, ToonNetDeep
from datasets.TinyImagenet import TinyImagenet
from datasets.Imagenet import Imagenet
from utils import montage
from constants import MODEL_DIR, IMG_DIR

batch_size = 32
nb_epoch = 20
nb_epoch_inner = 1
plot_while_train = True
f_dims = [64, 96, 160, 256, 416, 512]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mse'

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
# toon_net, encoder, decoder = ToonNet(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid',
#                                      num_res_layers=num_res_layers, merge_mode=merge_mode, f_dims=f_dims)
toon_net, encoder, decoder = ToonNetDeep(input_shape=data.get_dims(), batch_size=batch_size, out_activation='sigmoid',
                                     num_res_layers=num_res_layers, merge_mode=merge_mode, f_dims=f_dims)

# Name used for saving of model and outputs
net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}'.format(toon_net.name, f_dims, num_res_layers, merge_mode, loss,
                                                                       data.name)
print(net_name)

toon_net.summary()  # For debugging

# Define objective and solver
toon_net.compile(optimizer='adam', loss=loss)

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
        if plot_while_train and chunk_count % 10 == 0:
            # Test images
            decoded_imgs = toon_net.predict(X_test[:batch_size], batch_size=batch_size)
            montage(decoded_imgs[:16, :, :], os.path.join(IMG_DIR, 'Train: {}-{}'.format(net_name, chunk_count)))
        del X_train, Y_train, X_test, Y_test
        gc.collect()

    # Save the model after each epoch
    toon_net.save(os.path.join(MODEL_DIR, '{}-Epoch:{}-{}.h5'.format(net_name, e, nb_epoch)))

# Save the model
toon_net.save(os.path.join(MODEL_DIR, '{}.h5'.format(net_name)))

# Generate montage of test-images
X_test, Y_test = data.generator_test(batch_size=batch_size).next()
decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
montage(X_test[:25, :, :], os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(decoded_imgs[:25, :, :], os.path.join(IMG_DIR, '{}-Out'.format(net_name)))
montage(Y_test[:25, :, :], os.path.join(IMG_DIR, '{}-Y'.format(net_name)))
