from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from ToonNetAEGAN_normal_gan import VAEGAN
from constants import LOG_DIR
from datasets import imagenet
from preprocess import preprocess_finetune_train, preprocess_finetune_test
from utils import assign_from_checkpoint_fn, montage
import numpy as np

slim = tf.contrib.slim

# Setup
fine_tune = False
net_type = 'discriminator'
data = imagenet
num_layers = 5
model = VAEGAN(num_layers=num_layers, batch_size=64, data_size=data.SPLITS_TO_SIZES['train'], num_epochs=60)
TARGET_SHAPE = [224, 224, 3]
TEST_WHILE_TRAIN = False
NUM_CONV_TRAIN = 5
pre_trained_grad_weight = [0.25 * 0.25 ** i for i in range(NUM_CONV_TRAIN)]

CHECKPOINT = 'model.ckpt-100002'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_final/{}'.format(data.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_final/'.format(data.NAME, model.name,
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
            [img_train, edge_train, label_train] = provider.get(['image', 'edges', 'label'])

            # Pre-process data
            img_train, edge_train = preprocess_finetune_train(img_train, edge_train,
                                                              output_height=TARGET_SHAPE[0],
                                                              output_width=TARGET_SHAPE[1],
                                                              resize_side_min=224,
                                                              resize_side_max=256)

            # Make batches
            imgs_train, edges_train, labels_train = tf.train.batch(
                [img_train, edge_train, label_train],
                batch_size=model.batch_size, num_threads=8,
                capacity=4 * model.batch_size)

            if TEST_WHILE_TRAIN:
                # Get test-data
                test_set = data.get_split('test')
                provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
                [img_test, edge_test, label_test] = provider.get(['image', 'edges', 'label'])
                img_test, edge_test = preprocess_finetune_test(img_test, edge_test,
                                                               output_height=TARGET_SHAPE[0],
                                                               output_width=TARGET_SHAPE[1],
                                                               resize_side=224)
                imgs_test, edges_test, labels_test = tf.train.batch(
                    [img_test, edge_test, label_test],
                    batch_size=model.batch_size, num_threads=4)

        # Get predictions
        preds_train = model.classifier(imgs_train, edges_train, data.NUM_CLASSES, type=net_type,
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
        boundaries = [np.int64(num_train_steps * 0.33), np.int64(num_train_steps * 0.66)]
        values = [0.0002, 0.0001, 0.00005]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

        # Create training operation
        if fine_tune:
            grad_multipliers = {}
            var2train = []
            for i in range(NUM_CONV_TRAIN):
                vs = slim.get_variables_to_restore(include=['{}/conv_{}'.format(net_type, NUM_CONV_TRAIN - i)],
                                                   exclude=['discriminator/fully_connected'])
                vs = list(set(vs).intersection(tf.trainable_variables()))
                var2train += vs
                for v in vs:
                    grad_multipliers[v.op.name] = pre_trained_grad_weight[i]
            vs = slim.get_variables_to_restore(include=['fully_connected'], exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
            for v in vs:
                grad_multipliers[v.op.name] = 1.0
        else:
            var2train = tf.trainable_variables()
            grad_multipliers = None

        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, gradient_multipliers=grad_multipliers)
        print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        print(grad_multipliers)
        sys.stdout.flush()

        if TEST_WHILE_TRAIN:
            preds_test = model.classifier(imgs_test, edges_test, data.NUM_CLASSES, reuse=True,
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
        tf.image_summary('images/ground-truth', montage(imgs_train, 4, 4), max_images=1)

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
