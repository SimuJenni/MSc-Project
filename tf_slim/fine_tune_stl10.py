from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from ToonNetAEGAN import VAEGAN
from constants import LOG_DIR
from datasets import stl10
from preprocess import preprocess_toon_train, preprocess_toon_test
from utils import assign_from_checkpoint_fn
import numpy as np

slim = tf.contrib.slim

# Setup
fine_tune = True
net_type = 'discriminator'
data = stl10
num_layers = 4
model = VAEGAN(num_layers=num_layers, batch_size=128, data_size=data.SPLITS_TO_SIZES['train'], num_epochs=300)
TARGET_SHAPE = [96, 96, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)
TEST_WHILE_TRAIN = False
NUM_CONV_TRAIN = 3
pre_trained_grad_weight = 0.1

CHECKPOINT = 'model.ckpt-234302'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_andanothersetting/{}'.format(data.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_andanothersetting_do0.9/'.format(data.NAME, model.name,
                                                                                             net_type, NUM_CONV_TRAIN))
else:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        with tf.device('/cpu:0'):

            # Get the training dataset
            train_set = data.get_split('train')
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=8,
                                                                      common_queue_capacity=32 * model.batch_size,
                                                                      common_queue_min=4 * model.batch_size)
            [img_train, edge_train, toon_train, label_train] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Pre-process data
            img_train, edge_train, toon_train = preprocess_toon_train(img_train, edge_train, toon_train,
                                                                      output_height=TARGET_SHAPE[0],
                                                                      output_width=TARGET_SHAPE[1],
                                                                      resize_side_min=RESIZE_SIZE,
                                                                      resize_side_max=int(RESIZE_SIZE * 1.5))

            # Make batches
            imgs_train, edges_train, toons_train, labels_train = tf.train.batch(
                [img_train, edge_train, toon_train, label_train],
                batch_size=model.batch_size, num_threads=8,
                capacity=4 * model.batch_size)

            if TEST_WHILE_TRAIN:
                # Get test-data
                test_set = data.get_split('test')
                provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
                [img_test, edge_test, toon_test, label_test] = provider.get(['image', 'edges', 'cartoon', 'label'])
                img_test, edge_test, toon_test = preprocess_toon_test(img_test, edge_test, toon_test,
                                                                      output_height=TARGET_SHAPE[0],
                                                                      output_width=TARGET_SHAPE[1],
                                                                      resize_side=RESIZE_SIZE)
                imgs_test, edges_test, toons_test, labels_test = tf.train.batch(
                    [img_test, edge_test, toon_test, label_test],
                    batch_size=model.batch_size, num_threads=4)

        # Get predictions
        preds_train = model.classifier(imgs_train, edges_train, toons_train, data.NUM_CLASSES, type=net_type,
                                       fine_tune=fine_tune)

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
        num_train_steps = (data.SPLITS_TO_SIZES['train'] / model.batch_size) * model.num_ep
        boundaries = [np.int64(num_train_steps*0.25), np.int64(num_train_steps*0.5), np.int64(num_train_steps*0.75)]
        values = [0.001, 0.001 * 100. ** (-1. / 3.), 0.001 * 100 ** (-2. / 3.), 0.001 * 100 ** (-3. / 3.)]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

        # Create training operation
        trainable_scopes = ['{}/conv_{}'.format(net_type, num_layers-i) for i in range(NUM_CONV_TRAIN)]
        trainable_scopes += ['fully_connected']
        var2train = slim.get_variables_to_restore(include=trainable_scopes, exclude=['discriminator/fully_connected'])
        var2train = list(set(var2train).intersection(tf.trainable_variables()))
        pre_trained_vars = slim.get_variables(scope=net_type)
        grad_multipliers = {}
        for v in var2train:
            if v in pre_trained_vars:
                grad_multipliers[v.op.name] = pre_trained_grad_weight
            else:
                grad_multipliers[v.op.name] = 1.0

        print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        print('Pre-trained vars: {}'.format([v.op.name for v in pre_trained_vars]))
        print(grad_multipliers)
        sys.stdout.flush()

        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, gradient_multipliers=grad_multipliers)

        if TEST_WHILE_TRAIN:
            preds_test = model.classifier(imgs_test, edges_test, toons_test, data.NUM_CLASSES, reuse=True,
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

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                include=[net_type], exclude=['fully_connected', 'discriminator/fully_connected',
                                             ops.GraphKeys.GLOBAL_STEP])
            print('Variables to restore: {}'.format([v.op.name for v in variables_to_restore]))
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=60, save_interval_secs=600,
                            log_every_n_steps=100)
