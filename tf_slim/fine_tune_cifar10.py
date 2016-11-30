from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

from ToonNet import AEGAN2
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_images_toon, preprocess_images_toon_test
from utils import get_variables_to_train, assign_from_checkpoint_fn

slim = tf.contrib.slim

fine_tune = True
type = 'generator'
data = cifar10
model = AEGAN2(num_layers=4, batch_size=128, data_size=data.SPLITS_TO_SIZES['train'], num_epochs=200)
TARGET_SHAPE = [32, 32, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)

CHECKPOINT = 'model.ckpt-78002'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}/{}'.format(data.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}_notrainbn/'.format(data.NAME, model.name, type))
else:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Pre-process training data
        with tf.device('/cpu:0'):

            # Get the training dataset
            dataset = data.get_split('train')
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=8,
                common_queue_capacity=32 * model.batch_size,
                common_queue_min=4 * model.batch_size)
            [img_train, edge_train, toon_train, label_train] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Get some test-data
            test_set = data.get_split('test')
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
            [img_test, edge_test, toon_test, label_test] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Preprocess data
            img_train, edge_train, toon_train = preprocess_images_toon(img_train, edge_train, toon_train,
                                                                       output_height=TARGET_SHAPE[0],
                                                                       output_width=TARGET_SHAPE[1],
                                                                       resize_side_min=RESIZE_SIZE,
                                                                       resize_side_max=int(RESIZE_SIZE * 1.5))
            img_test, edge_test, toon_test = preprocess_images_toon_test(img_test, edge_test, toon_test,
                                                                         output_height=TARGET_SHAPE[0],
                                                                         output_width=TARGET_SHAPE[1],
                                                                         resize_side=RESIZE_SIZE)

            # Make batches
            imgs_train, edges_train, toons_train, labels_train = tf.train.batch(
                [img_train, edge_train, toon_train, label_train],
                batch_size=model.batch_size, num_threads=8,
                capacity=4 * model.batch_size)
            imgs_test, edges_test, toons_test, labels_test = tf.train.batch(
                [img_test, edge_test, toon_test, label_test],
                batch_size=model.batch_size, num_threads=4)

        # Create the model
        preds_train = model.classifier2(imgs_train, edges_train, toons_train, data.NUM_CLASSES, finetune=fine_tune)
        preds_test = model.classifier2(imgs_test, edges_test, toons_test, data.NUM_CLASSES, reuse=True, training=False,
                                      finetune=fine_tune)

        # Define the loss
        train_loss = slim.losses.softmax_cross_entropy(preds_train, slim.one_hot_encoding(labels_train, data.NUM_CLASSES))
        total_train_loss = slim.losses.get_total_loss()
        test_loss = slim.losses.softmax_cross_entropy(preds_test, slim.one_hot_encoding(labels_test, data.NUM_CLASSES))

        # Compute predicted label for accuracy
        preds_train = tf.argmax(preds_train, 1)
        preds_test = tf.argmax(preds_test, 1)

        # Handle dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_train_loss = control_flow_ops.with_dependencies([updates], total_train_loss)

        # Define learning parameters
        num_train_steps = (data.SPLITS_TO_SIZES['train'] / model.batch_size) * model.num_ep
        learning_rate = tf.select(tf.python.math_ops.greater(global_step, num_train_steps / 2),
                                  0.001 - 0.001 * (2*tf.cast(global_step, tf.float32)/num_train_steps-1.0), 0.001)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9)

        # Gather all summaries.
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/training loss', train_loss)
        tf.scalar_summary('losses/test loss', test_loss)
        tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))
        tf.scalar_summary('accuracy/test', slim.metrics.accuracy(preds_test, labels_test))

        # Create training operation
        if fine_tune:
            var2train = get_variables_to_train(trainable_scopes='fully_connected')
        else:
            var2train = get_variables_to_train()
        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in var2train:
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fully_connected', ops.GraphKeys.GLOBAL_STEP])
            slim.get_or_create_global_step()
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=60, save_interval_secs=600,
                            log_every_n_steps=100)
