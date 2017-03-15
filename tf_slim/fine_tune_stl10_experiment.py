from __future__ import print_function

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

from ToonNet_VGG import VAEGAN
from constants import LOG_DIR
from datasets import stl10
from preprocess import preprocess_toon_train, preprocess_finetune_train
from utils import assign_from_checkpoint_fn, montage_tf

slim = tf.contrib.slim


def fine_tune_model(data, num_layers, num_conv_train, target_shape, checkpoint, train_set_id, batch_size, num_epochs,
                    net_type='discriminator', fine_tune=True):
    model = VAEGAN(num_layers=num_layers, batch_size=batch_size)
    model_path = os.path.join(LOG_DIR, '{}_{}_exp6/{}'.format(data.NAME, model.name, checkpoint))
    if fine_tune:
        save_dir = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_exp6/'.format(data.NAME,
                                                                                       model.name,
                                                                                       net_type,
                                                                                       num_conv_train))
    else:
        save_dir = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

    sess = tf.Session()
    tf.logging.set_verbosity(tf.logging.DEBUG)

    g = tf.Graph()
    with sess.as_default():
        with g.as_default():
            global_step = slim.create_global_step()

            with tf.device('/cpu:0'):
                # Get the training dataset
                train_set = data.get_split(train_set_id)
                provider_train = slim.dataset_data_provider.DatasetDataProvider(train_set,
                                                                                num_readers=8,
                                                                                common_queue_capacity=32 * batch_size,
                                                                                common_queue_min=4 * batch_size)
                [img_train, label_train, edge_train] = provider_train.get(['image', 'label', 'edges'])

                # Pre-process data
                img_train = preprocess_finetune_train(img_train, output_height=target_shape[0],
                                                      output_width=target_shape[1], augment_color=False,
                                                      resize_side_min=96, resize_side_max=120)
                # img_train, edge_train, _ = preprocess_toon_train(img_train, edge_train, img_train,
                #                                                  output_height=target_shape[0],
                #                                                  output_width=target_shape[1],
                #                                                  resize_side_min=96,
                #                                                  resize_side_max=120)
                # Make batches
                imgs_train, labels_train, edges_train = tf.train.batch([img_train, label_train, edge_train],
                                                                       batch_size=batch_size,
                                                                       num_threads=8, capacity=4 * batch_size)

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
            num_train_steps = (data.SPLITS_TO_SIZES[train_set_id] / batch_size) * num_epochs
            boundaries = [np.int64(num_train_steps * 0.25), np.int64(num_train_steps * 0.5),
                          np.int64(num_train_steps * 0.75)]
            values = [0.0002, 0.0001, 0.00005, 0.000025]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=values)

            # Define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, epsilon=1e-5)

            # Create training operation
            if fine_tune:
                var2train = []
                for i in range(num_conv_train):
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

            train_op = slim.learning.create_train_op(total_train_loss, optimizer, variables_to_train=var2train,
                                                     global_step=global_step, summarize_gradients=True)
            print('Trainable vars: {}'.format([v.op.name for v in tf.trainable_variables()]))
            print('Variables to train: {}'.format([v.op.name for v in var2train]))
            sys.stdout.flush()

            # Gather all summaries
            for variable in slim.get_model_variables():
                tf.histogram_summary(variable.op.name, variable)
            tf.scalar_summary('learning rate', learning_rate)
            tf.scalar_summary('losses/training loss', train_loss)
            tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))
            tf.image_summary('images/ground-truth', montage_tf(imgs_train, 4, 4), max_images=1)
            tf.histogram_summary('lables', labels_train)
            tf.histogram_summary('predictions', preds_train)

            # Handle initialisation
            init_fn = None
            if fine_tune:
                # Specify the layers of your model you want to exclude
                variables_to_restore = slim.get_variables_to_restore(
                    include=[net_type], exclude=['fully_connected', 'discriminator/fully_connected',
                                                 ops.GraphKeys.GLOBAL_STEP])
                print('Variables to restore: {}'.format([v.op.name for v in variables_to_restore]))
                init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore, ignore_missing_vars=True)

            # Start training
            slim.learning.train(train_op, save_dir,
                                init_fn=init_fn, number_of_steps=num_train_steps,
                                save_summaries_secs=60, save_interval_secs=600,
                                log_every_n_steps=100)


fine_tune_model(stl10, 4, 0, [96, 96, 3], 'model.ckpt-150000', 'train', 64, 200, fine_tune=True, net_type='discriminator')
