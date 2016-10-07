import os

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from ToonNet import ToonNetDeep
from constants import MODEL_DIR, IMG_DIR, LOG_DIR
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32
nb_epoch = 10
samples_per_epoch = 200000
plot_while_train = True
f_dims = [64, 96, 160, 256, 416, 512]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mae'
train_im_size = (128, 128, 3)

# Get the data-set object
data = Imagenet(train_im_size)

# Load the net
toon_net, encoder, decoder = ToonNetDeep(input_shape=train_im_size, batch_size=batch_size, out_activation='tanh',
                                         num_res_layers=num_res_layers, merge_mode=merge_mode, f_dims=f_dims)

# Name used for saving of model and outputs
net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}'.format(toon_net.name, f_dims, num_res_layers, merge_mode,
                                                                  loss,
                                                                  data.name)
print('Training network: {}'.format(net_name))

# Define objective and solver
opt = Adam(lr=0.0002, beta_1=0.5)
toon_net.compile(optimizer=opt, loss=loss)

X_test, Y_test = data.generator_test(batch_size=batch_size).next()

# Training
toon_net.fit_generator(data.train_batch_generator(batch_size=batch_size),
                       samples_per_epoch=samples_per_epoch,
                       nb_epoch=nb_epoch,
                       validation_data=data.test_batch_generator(batch_size=batch_size),
                       nb_val_samples=30000,
                       nb_worker=4,
                       callbacks=[ModelCheckpoint(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name))),
                                  TensorBoard(log_dir=LOG_DIR)])

# Generate montage of test-images
decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out'.format(net_name)))
montage(X_test[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(Y_test[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y'.format(net_name)))

"""

# Training
for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))
    num_samples = 0

    for X_train, Y_train, X_val, Y_val in data.generator(batch_size=batch_size):
        toon_net.fit(X_train, Y_train,
                     batch_size=batch_size,
                     nb_epoch=1,
                     validation_data=(X_val, Y_val),
                     callbacks=[ModelCheckpoint(os.path.join(MODEL_DIR, '{}.hdf5'.format(net_name))),
                                TensorBoard(log_dir=LOG_DIR)])
        num_samples += X_train.shape[0]
        del X_train, X_val, Y_train, Y_val
        gc.collect()
        if num_samples>samples_per_epoch:
            break

    # Save the model
    toon_net.save(os.path.join(MODEL_DIR, '{}.h5'.format(net_name)))

    # Generate montage of test-images
    decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
    montage(decoded_imgs[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out'.format(net_name)))

montage(X_test[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(Y_test[:49, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y'.format(net_name)))

"""
