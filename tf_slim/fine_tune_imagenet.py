from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNet_Alex3 import VAEGAN
from constants import LOG_DIR
from datasets import imagenet
from preprocess import preprocess_finetune_test, preprocess_imagenet
from utils import assign_from_checkpoint_fn, montage_tf
import numpy as np
from constants import IMAGENET_TF_DATADIR

slim = tf.contrib.slim

# Setup
fine_tune = True
net_type = 'discriminator'
data = imagenet
num_layers = 5
model = VAEGAN(num_layers=num_layers, batch_size=256)
TARGET_SHAPE = [224, 224, 3]
TEST_WHILE_TRAIN = False
NUM_CONV_TRAIN = 1
num_epochs = 60

CHECKPOINT = 'model.ckpt-600542'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_final/{}'.format(data.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_final/'.format(data.NAME, model.name,
                                                                                 net_type, NUM_CONV_TRAIN))
else:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # with tf.device('/cpu:0'):

        # Get the training dataset
        train_set = data.get_split('train', dataset_dir=IMAGENET_TF_DATADIR)
        provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=2,
                                                                  common_queue_capacity=4 * model.batch_size,
                                                                  common_queue_min=128)
        [img_train, label_train] = provider.get(['image', 'label'])
        label_train -= data.LABEL_OFFSET

        # Pre-process data
        img_train = preprocess_imagenet(img_train, output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1])

        # Make batches
        imgs_train, labels_train = tf.train.batch([img_train, label_train], batch_size=model.batch_size,
                                                  num_threads=16, capacity=2*model.batch_size)

        if TEST_WHILE_TRAIN:
            # Get test-data
            test_set = data.get_split('test')
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
            [img_test, label_test] = provider.get(['image', 'label'])
            label_test -= data.LABEL_OFFSET
            img_test = preprocess_finetune_test(img_test,
                                                output_height=TARGET_SHAPE[0],
                                                output_width=TARGET_SHAPE[1],
                                                resize_side=128)
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=model.batch_size,
                                                    num_threads=4)

        # Get predictions
        preds_train = model.classifier(imgs_train, None, data.NUM_CLASSES, type=net_type, fine_tune=fine_tune)

        # Define the loss
        loss_scope = 'train_loss'
        train_loss = slim.losses.softmax_cross_entropy(preds_train,
                                                       slim.one_hot_encoding(labels_train, data.NUM_CLASSES),
                                                       scope=loss_scope)
        train_losses = slim.losses.get_losses(loss_scope)
        train_losses += slim.losses.get_regularization_losses(loss_scope)
        total_train_loss = math_ops.add_n(train_losses, name='total_train_loss')

        # Compute predicted label for accuracy
        preds_train = tf.argmax(preds_train, 1)

        # Handle dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_train_loss = control_flow_ops.with_dependencies([updates], total_train_loss)

        # Define learning parameters
        num_train_steps = (data.SPLITS_TO_SIZES['train'] / model.batch_size) * num_epochs
        boundaries = [np.int64(num_train_steps * 0.25), np.int64(num_train_steps * 0.5),
                      np.int64(num_train_steps * 0.75)]
        values = [0.001, 0.0005, 0.0002, 0.0001]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

        # Create training operation
        if fine_tune:
            var2train = []
            for i in range(NUM_CONV_TRAIN):
                vs = slim.get_variables_to_restore(include=['{}/conv_{}'.format(net_type, 5 - i)],
                                                   exclude=['discriminator/fully_connected'])
                vs = list(set(vs).intersection(tf.trainable_variables()))
                var2train += vs
            vs = slim.get_variables_to_restore(include=['fully_connected'], exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
        else:
            var2train = tf.trainable_variables()

        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step)
        print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
        print('Variables to train: {}'.format([v.op.name for v in var2train]))

        if TEST_WHILE_TRAIN:
            preds_test = model.classifier(imgs_test, None, data.NUM_CLASSES, reuse=True,
                                          training=False, fine_tune=fine_tune, type=net_type)
            test_loss = slim.losses.softmax_cross_entropy(preds_test,
                                                          slim.one_hot_encoding(labels_test, data.NUM_CLASSES))
            preds_test = tf.argmax(preds_test, 1)
            tf.scalar_summary('accuracy/test', slim.metrics.accuracy(preds_test, labels_test))
            tf.scalar_summary('losses/test loss', test_loss)

        # Gather all summaries
        for variable in slim.get_model_variables():
            tf.histogram_summary(variable.op.name, variable)
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/training loss', train_loss)
        tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))
        tf.image_summary('images/ground-truth', montage_tf(imgs_train, 4, 4), max_images=1)

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            var2restore = []
            for i in range(num_layers-NUM_CONV_TRAIN):
                vs = slim.get_variables_to_restore(include=['{}/conv_{}'.format(net_type, i + 1)],
                                                   exclude=['discriminator/fully_connected'])
                var2restore += vs
            print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, var2restore, ignore_missing_vars=True)

        # Start training
        sys.stdout.flush()
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=300, save_interval_secs=600,
                            log_every_n_steps=100)
