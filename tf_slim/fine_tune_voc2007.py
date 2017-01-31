from __future__ import print_function

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from ToonNet_Alex3 import VAEGAN
from constants import LOG_DIR
from datasets import voc, imagenet
from preprocess import preprocess_voc
from utils import assign_from_checkpoint_fn, montage_tf

slim = tf.contrib.slim

# Setup
fine_tune = True
net_type = 'discriminator'
data = voc
num_layers = 5
model = VAEGAN(num_layers=num_layers, batch_size=128)
TARGET_SHAPE = [224, 224, 3]
num_ep = 400
TEST_WHILE_TRAIN = False
NUM_CONV_TRAIN = 3
TRAIN_SET = 'trainval'
TEST_SET = 'test'
pre_trained_grad_weight = [0.5 * 0.5 ** i for i in range(NUM_CONV_TRAIN)]

CHECKPOINT = 'model.ckpt-600542'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_final/{}'.format(imagenet.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_final_{}_imnet/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, TRAIN_SET))
else:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # with tf.device('/cpu:0'):

        # Get the training dataset
        train_set = data.get_split(TRAIN_SET)
        provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=2,
                                                                  common_queue_capacity=2 * model.batch_size,
                                                                  common_queue_min=128)
        [img_train, label_train] = provider.get(['image', 'label'])

        # Pre-process data
        img_train = preprocess_voc(img_train, output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1],
                                   augment_color=True, aspect_ratio_range=[0.5, 1.5], area_range=[0.2, 1.0])

        # Make batches
        imgs_train, labels_train = tf.train.batch([img_train, label_train],
                                                  batch_size=model.batch_size, num_threads=32,
                                                  capacity=model.batch_size)

        if TEST_WHILE_TRAIN:
            # Get test-data
            test_set = data.get_split(TEST_SET)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test] = provider.get(['image', 'label'])
            img_test = preprocess_voc(img_test, output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1])
            imgs_test, labels_test = tf.train.batch(
                [img_test, label_test],
                batch_size=model.batch_size, num_threads=1)

        # Get predictions
        preds_train = model.classifier(imgs_train, None, data.NUM_CLASSES, type=net_type, fine_tune=fine_tune)

        # Define the loss
        loss_scope = 'train_loss'
        train_loss = slim.losses.sigmoid_cross_entropy(preds_train, labels_train, scope=loss_scope)
        train_losses = slim.losses.get_losses(loss_scope)
        train_losses += slim.losses.get_regularization_losses(loss_scope)
        total_train_loss = math_ops.add_n(train_losses, name='total_train_loss')

        # Handle dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_train_loss = control_flow_ops.with_dependencies([updates], total_train_loss)

        # Define learning parameters
        num_train_steps = (data.SPLITS_TO_SIZES[TRAIN_SET] / model.batch_size) * num_ep
        boundaries = [np.int64(num_train_steps * 0.25), np.int64(num_train_steps * 0.5),
                      np.int64(num_train_steps * 0.75)]
        values = [0.0002, 0.0001, 0.00005, 0.000025]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

        # Create training operation
        if fine_tune:
            grad_multipliers = {}
            var2train = []
            for i in range(NUM_CONV_TRAIN):
                vs = slim.get_variables_to_restore(include=['{}/conv_{}'.format(net_type, 5 - i)],
                                                   exclude=['discriminator/fully_connected', 'discriminator/fc1'])
                vs = list(set(vs).intersection(tf.trainable_variables()))
                var2train += vs
                for v in vs:
                    grad_multipliers[v.op.name] = pre_trained_grad_weight[i]
            vs = slim.get_variables_to_restore(include=['fully_connected'], exclude=['discriminator/fully_connected',
                                                                                     'discriminator/fc1'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
            for v in vs:
                grad_multipliers[v.op.name] = 1.0
        else:
            var2train = tf.trainable_variables()
            grad_multipliers = None

        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, gradient_multipliers=grad_multipliers,
                                                 summarize_gradients=True)
        print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        print(grad_multipliers)

        if TEST_WHILE_TRAIN:
            preds_test = model.classifier(imgs_test, None, data.NUM_CLASSES, reuse=True,
                                          training=False, fine_tune=fine_tune, type=net_type)
            test_loss = slim.losses.sigmoid_cross_entropy(preds_test, labels_test)
            tf.scalar_summary('losses/test loss', test_loss)

        # Gather all summaries
        for variable in slim.get_model_variables():
            tf.histogram_summary(variable.op.name, variable)
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/training loss', train_loss)
        tf.image_summary('images/ground-truth', montage_tf(imgs_train, 4, 4), max_images=1)

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                include=[net_type], exclude=['fully_connected', 'discriminator/fully_connected', 'discriminator/fc1',
                                             ops.GraphKeys.GLOBAL_STEP])
            print('Variables to restore: {}'.format([v.op.name for v in variables_to_restore]))
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        sys.stdout.flush()
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=60, save_interval_secs=180,
                            log_every_n_steps=100)
