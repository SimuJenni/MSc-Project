from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from ToonNet_AlexV2 import VAEGAN
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
model = VAEGAN(num_layers=num_layers, batch_size=32)
TARGET_SHAPE = [224, 224, 3]
num_ep = 500
NUM_CONV_TRAIN = 5
TRAIN_SET = 'trainval'
num_preprocess_threads = 8

CHECKPOINT = 'model.ckpt-900811'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_final/{}'.format(imagenet.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_lrelu_{}/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, TRAIN_SET))
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
            train_set = data.get_split(TRAIN_SET)
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=1,
                                                                      common_queue_capacity=4 * model.batch_size,
                                                                      common_queue_min=model.batch_size)
            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                [img_train, label_train] = provider.get(['image', 'label'])

                # Pre-process data
                img_train = preprocess_voc(img_train, output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1],
                                           augment_color=True)
                images_and_labels.append([img_train, label_train])

            # Make batches
            imgs_train, labels_train = tf.train.batch_join(
                images_and_labels,
                batch_size=model.batch_size,
                capacity=num_preprocess_threads * model.batch_size)

        # Get predictions
        preds_train = model.build_classifier(imgs_train, data.NUM_CLASSES)

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
        learning_rate = tf.train.polynomial_decay(0.0002, global_step, num_train_steps, end_learning_rate=0.0)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

        # Create training operation
        if fine_tune:
            # grad_multipliers = {}
            var2train = []
            for i in range(NUM_CONV_TRAIN):
                vs = slim.get_variables_to_restore(include=['{}/conv_{}'.format(net_type, 5 - i)],
                                                   exclude=['discriminator/fully_connected'])
                vs = list(set(vs).intersection(tf.trainable_variables()))
                var2train += vs
            vs = slim.get_variables_to_restore(include=['fully_connected'],
                                               exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
        else:
            var2train = tf.trainable_variables()
            grad_multipliers = None

        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, summarize_gradients=True)
        print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
        print('Variables to train: {}'.format([v.op.name for v in var2train]))

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
                include=[net_type], exclude=['fully_connected', 'discriminator/fully_connected',
                                             ops.GraphKeys.GLOBAL_STEP])
            print('Variables to restore: {}'.format([v.op.name for v in variables_to_restore]))
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        sys.stdout.flush()
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=60, save_interval_secs=180,
                            log_every_n_steps=100)
