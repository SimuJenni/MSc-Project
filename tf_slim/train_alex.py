from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops

from alexnet_v2 import alexnet_v2, alexnet_v2_arg_scope
from datasets import imagenet
from preprocess import preprocess_image
from utils import get_variables_to_train

slim = tf.contrib.slim

FINE_TUNE = False
BATCH_SIZE = 256
NUM_CLASSES = 1000
NUM_EP = 90
IM_SHAPE = [224, 224, 3]
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
LOG_DIR = '/data/cvg/simon/data/logs/alex_net_run2/'  # TODO: specify log-dir
TEST_WHILE_TRAIN = False

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Pre-process training data
        with tf.device('/cpu:0'):
            # Get the training dataset
            dataset = imagenet.get_split('train', dataset_dir=DATA_DIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=8,
                                                                      common_queue_capacity=32 * BATCH_SIZE,
                                                                      common_queue_min=8 * BATCH_SIZE)
            [img_train, label] = provider.get(['image', 'label'])

            # Pre-process images
            img_train = preprocess_image(img_train, is_training=True, output_height=IM_SHAPE[0],
                                         output_width=IM_SHAPE[1])
            # Make batches
            imgs_train, labels_train = tf.train.batch([img_train, label], batch_size=BATCH_SIZE, num_threads=8,
                                                      capacity=8 * BATCH_SIZE)

        # Create the model
        with slim.arg_scope(alexnet_v2_arg_scope()):
            predictions = alexnet_v2(imgs_train, is_training=True)

        # Define the loss
        train_loss = slim.losses.softmax_cross_entropy(predictions, slim.one_hot_encoding(labels_train, NUM_CLASSES))
        total_loss = slim.losses.get_total_loss()

        # Compute predictions for accuracy computation
        preds_train = tf.argmax(predictions, 1)

        # Define learning rate
        num_train_steps = (imagenet.SPLITS_TO_SIZES['train'] / BATCH_SIZE) * NUM_EP
        boundaries = [np.int64(num_train_steps*0.25), np.int64(num_train_steps*0.5), np.int64(num_train_steps*0.75)]
        values = [0.01, 0.01*250.**(-1./3.), 0.01*250**(-2./3.),  0.01*250.**(-1.)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # Define optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

        # Gather all summaries.
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/train loss', train_loss)
        tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))

        # Create training operation
        var2train = get_variables_to_train()
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in var2train:
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        # Start training.
        slim.learning.train(train_op, LOG_DIR,
                            number_of_steps=num_train_steps,
                            save_summaries_secs=120,
                            save_interval_secs=1200,
                            log_every_n_steps=100)
