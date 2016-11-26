from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from ToonNet import AEGAN2
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_images_toon, preprocess_images_toon_test
from utils import get_variables_to_train, assign_from_checkpoint_fn

slim = tf.contrib.slim

fine_tune = True
type = 'generator'
data = cifar10
model = AEGAN2(num_layers=4, batch_size=256, data_size=data.SPLITS_TO_SIZES['train'], num_epochs=50)
TARGET_SHAPE = [32, 32, 3]

CHECKPOINT = 'model.ckpt-19500'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}/{}'.format(data.NAME, model.name, CHECKPOINT))
if fine_tune:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_finetune_{}/'.format(data.NAME, model.name, type))
else:
    SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

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
                num_readers=16,
                common_queue_capacity=32 * model.batch_size,
                common_queue_min=4 * model.batch_size)
            [img_train, edge_train, toon_train, label_train] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Preprocess data
            img_train, edge_train, toon_train = preprocess_images_toon(img_train, edge_train, toon_train,
                                                                       output_height=TARGET_SHAPE[0],
                                                                       output_width=TARGET_SHAPE[1],
                                                                       resize_side_min=data.MIN_SIZE,
                                                                       resize_side_max=int(data.MIN_SIZE * 1.5))
            # Make batches
            imgs_train, edges_train, toons_train, labels_train = tf.train.batch(
                [img_train, edge_train, toon_train, label_train],
                batch_size=model.batch_size, num_threads=16,
                capacity=4 * model.batch_size)

            # Get some test-data
            test_set = data.get_split('test')
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, shuffle=False, num_readers=16)
            [img_test, edge_test, toon_test, label_test] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Preprocess data
            img_test, edge_test, toon_test = preprocess_images_toon_test(img_test, edge_test, toon_test,
                                                                         output_height=TARGET_SHAPE[0],
                                                                         output_width=TARGET_SHAPE[1],
                                                                         resize_side=data.MIN_SIZE)
            imgs_test, edges_test, toons_test, labels_test = tf.train.batch(
                [img_train, edge_train, toon_train, label_test],
                batch_size=model.batch_size, num_threads=16)

        # Create the model
        preds_train = model.classifier(imgs_train, edges_train, toons_train, data.NUM_CLASSES)
        preds_test = model.classifier(imgs_test, edges_test, toons_test, data.NUM_CLASSES, reuse=True, training=False)

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

        # Define learning rate
        decay_steps = int(data.SPLITS_TO_SIZES['train'] / model.batch_size * 2.0)
        learning_rate = tf.train.exponential_decay(0.01,
                                                   global_step,
                                                   decay_steps,
                                                   0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Gather all summaries.
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/training loss', train_loss)
        tf.scalar_summary('losses/test loss', test_loss)
        tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))
        tf.scalar_summary('accuracy/test', slim.metrics.accuracy(preds_test, labels_test))

        # Define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1.0, momentum=0.9, decay=0.9)

        # Create training operation
        if fine_tune:
            var2train = get_variables_to_train(trainable_scopes='fully_connected')
        else:
            var2train = get_variables_to_train()
        train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, summarize_gradients=True)

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fully_connected'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        num_train_steps = data.SPLITS_TO_SIZES['train'] / model.batch_size * model.num_ep
        slim.learning.train(train_op, SAVE_DIR,
                            init_fn=init_fn, number_of_steps=num_train_steps,
                            save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
