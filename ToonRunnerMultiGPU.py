import os
from DataGenerator import ImageDataGenerator

from keras.objectives import MAE

from ToonNet import ToonNet
from constants import MODEL_DIR, IMG_DIR, LOG_DIR
from datasets.TinyImagenet import TinyImagenet
from datasets.Imagenet import Imagenet
from utils import montage

import tensorflow as tf
from keras import backend as K

batch_size = 32
nb_epoch = 10
samples_per_epoch = 20000
plot_while_train = True
f_dims = [64, 96, 160, 256, 512]
num_res_layers = 8
merge_mode = 'sum'
loss = 'mae'

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

with tf.device('/cpu:0'):
    K.manual_variable_initialization(True)

    # Load the net
    model, encoder, decoder = ToonNet(input_shape=data.dims, batch_size=batch_size, out_activation='tanh',
                                      num_res_layers=num_res_layers, merge_mode=merge_mode, f_dims=f_dims)

# replica 0
with tf.device('/gpu:0'):
    output_0 = model.output  # all ops in the replica will live on GPU:0

# replica 1
with tf.device('/gpu:1'):
    output_1 = model.output  # all ops in the replica will live on GPU:1

# merge outputs on CPU
with tf.device('/cpu:0'):
    preds = 0.5 * (output_0 + output_1)

    # Name used for saving of model and outputs
    net_name = '{}-f_dims:{}-NRes:{}-Merge:{}-Loss:{}-Data:{}'.format(model.name, f_dims, num_res_layers, merge_mode,
                                                                      loss,
                                                                      data.name)
    print('Training network: {}'.format(net_name))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        K.set_session(sess)
        init_op = tf.initialize_all_variables()

        y = tf.placeholder(tf.float32, shape=(None,)+data.dims, name='y')

        # reconstruction loss objective
        recon_loss = tf.reduce_mean(MAE(y, preds))

        # apply regularizers if any
        if model.regularizers:
            total_loss = recon_loss * 1.  # copy tensor
            for regularizer in model.regularizers:
                total_loss = regularizer(total_loss)
        else:
            total_loss = recon_loss

        # set up TF optimizer
        optimizer = tf.train.AdamOptimizer(0.0005)
        train_step = optimizer.minimize(total_loss)
        sess.run(init_op)

        step = 0
        for epoch in range(nb_epoch):
            print("Epoch {} / {}".format(epoch + 1, nb_epoch))
            for X_batch, Y_batch in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):
                feed_dict = {model.input: X_batch,
                             y: Y_batch,
                             K.learning_phase(): 1}
                _, train_loss = sess.run([train_step, total_loss],
                                               feed_dict=feed_dict)
                step += 1
            print("Step: %d," % step,
                  " Epoch: %2d," % (epoch + 1),
                  " Cost: %.4f," % train_loss)


# Generate montage of sample-images
sample_size = 64
X_test, Y_test = data.get_sample_dir(sample_size)
decoded_imgs = model.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out'.format(net_name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X'.format(net_name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y'.format(net_name)))
