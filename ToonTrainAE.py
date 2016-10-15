import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE, ToonAE2
from constants import MODEL_DIR, IMG_DIR, LOG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
nb_epoch = 1
samples_per_epoch = 1200000
f_dims = [64, 128, 256, 512, 1024]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mse'
l_rate = 0.0002

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the net
toonAE = ToonAE2(input_shape=data.dims, batch_size=batch_size, out_activation='tanh', num_res_layers=num_res_layers,
                f_dims=f_dims)
toonAE.summary()

# Name used for saving of model and outputs
net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}-LRate:{}'.format(toonAE.name, f_dims, num_res_layers,
                                                                           merge_mode, loss, data.name, l_rate)
print('Training network: {}'.format(net_name))

# Define objective and solver
opt = Adam(lr=l_rate, beta_1=0.5)
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
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(net_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(net_name)))