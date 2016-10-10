import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE
from constants import MODEL_DIR, IMG_DIR, LOG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
nb_epoch = 2
samples_per_epoch = 500000
f_dims = [64, 96, 160, 256, 512]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mae'
l_rate = 0.001

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator(horizontal_flip=True)

# Load the net
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size, out_activation='tanh', num_res_layers=num_res_layers,
                merge_mode=merge_mode, f_dims=f_dims)
toonAE.summary()

# Name used for saving of model and outputs
net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}-LRate:{}'.format(toonAE.name, f_dims, num_res_layers,
                                                                           merge_mode, loss, data.name, l_rate)
print('Training network: {}'.format(net_name))

# Define objective and solver
opt = Adam(lr=l_rate)
toonAE.compile(optimizer=opt, loss=loss)

# Training
history = toonAE.fit_generator(datagen.flow_from_directory(data.train_dir, batch_size=batch_size),
                               samples_per_epoch=samples_per_epoch,
                               nb_epoch=nb_epoch,
                               validation_data=datagen.flow_from_directory(data.val_dir, batch_size=batch_size),
                               nb_val_samples=32000,
                               nb_worker=4,
                               callbacks=[ModelCheckpoint(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name))),
                                          TensorBoard(log_dir=LOG_DIR, histogram_freq=1)])

# Save the model
toonAE.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name)))

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size).next()
decoded_imgs = toonAE.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out'.format(net_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y'.format(net_name)))