import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNet_VGGv2 import VAEGAN
from constants import LOG_DIR
from datasets import stl10
from preprocess import preprocess_finetune_train
from tf_slim.utils import get_variables_to_train
from utils import montage_tf

slim = tf.contrib.slim

# Setup training parameters
data = stl10
TRAIN_SET_NAME = 'train_unlabeled'
TEST_SET_NAME = 'test'
model = VAEGAN(num_layers=4, batch_size=500)
num_epochs = 200
TARGET_SHAPE = [64, 64, 3]
LR = 0.0002
SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_add_gaussian_noise/'.format(data.NAME, model.name))
NUM_IMG_SUMMARY = 6

tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.Session()
g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # with tf.device('/cpu:0'):

        # Get the training dataset
        train_set = data.get_split(TRAIN_SET_NAME)
        provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=8,
                                                                  common_queue_capacity=32 * model.batch_size,
                                                                  common_queue_min=4 * model.batch_size)
        [img_train] = provider.get(['image'])

        # Preprocess data
        img_train = preprocess_finetune_train(img_train,
                                              output_height=TARGET_SHAPE[0],
                                              output_width=TARGET_SHAPE[1],
                                              resize_side_min=96,
                                              resize_side_max=96)
        # Make batches
        imgs_train = tf.train.batch([img_train],
                                    batch_size=model.batch_size, num_threads=8,
                                    capacity=4 * model.batch_size)

        # Get labels for discriminator training
        labels_disc = model.disc_labels()

        # Create the model
        num_train_steps = (data.SPLITS_TO_SIZES[TRAIN_SET_NAME] / model.batch_size) * num_epochs
        disc_out, noise_imgs = model.experiment_net(imgs_train, num_train_steps)

        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        dL_disc = slim.losses.softmax_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope, weight=1.0)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5, epsilon=1e-6)

        # Handle summaries
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.image_summary('images/img', montage_tf(imgs_train, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
        tf.image_summary('images/noise_img', montage_tf(noise_imgs, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)

        # Discriminator training operation
        scopes_disc = 'discriminator'
        vars2train_disc = get_variables_to_train(trainable_scopes=scopes_disc)
        train_op_disc = slim.learning.create_train_op(disc_loss, optimizer, variables_to_train=vars2train_disc,
                                                      global_step=global_step, summarize_gradients=False)

        # Start training
        slim.learning.train(train_op_disc,
                            SAVE_DIR,
                            save_summaries_secs=300,
                            save_interval_secs=3000,
                            log_every_n_steps=100,
                            number_of_steps=num_train_steps)
