import os

from ToonNet import ToonNetDeep
from constants import MODEL_DIR, IMG_DIR
from datasets.Imagenet import Imagenet
from datasets.TinyImagenet import TinyImagenet
from utils import montage

batch_size = 32
nb_epoch = 20
nb_epoch_inner = 1
plot_while_train = True
f_dims = [64, 96, 160, 256, 416, 512]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mae'
train_im_size = (128, 128, 3)

# Get the data-set object
data = Imagenet(train_im_size)

# Load the net
toon_net, encoder, decoder = ToonNetDeep(input_shape=train_im_size, batch_size=batch_size, out_activation='sigmoid',
                                         num_res_layers=num_res_layers, merge_mode=merge_mode, f_dims=f_dims)

# Name used for saving of model and outputs
net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}'.format(toon_net.name, f_dims, num_res_layers, merge_mode,
                                                                  loss,
                                                                  data.name)
print(net_name)
toon_net.summary()  # For debugging

# Define objective and solver
toon_net.compile(optimizer='adam', loss=loss)

# Training
toon_net.fit_generator(data.train_batch_generator(batch_size=batch_size),
                       samples_per_epoch=data.num_train,
                       nb_epoch=nb_epoch,
                       validation_data=data.test_batch_generator(batch_size=batch_size),
                       nb_val_samples=data.num_val)

# Save the model
toon_net.save(os.path.join(MODEL_DIR, '{}.h5'.format(net_name)))

# Generate montage of test-images
X_test, Y_test = data.generator_test(batch_size=batch_size).next()
decoded_imgs = toon_net.predict(X_test, batch_size=batch_size)
montage(X_test[:25, :, :], os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(decoded_imgs[:25, :, :], os.path.join(IMG_DIR, '{}-Out'.format(net_name)))
montage(Y_test[:25, :, :], os.path.join(IMG_DIR, '{}-Y'.format(net_name)))
