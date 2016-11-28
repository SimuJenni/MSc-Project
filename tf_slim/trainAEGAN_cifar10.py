import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNet import AEGAN2
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_images_toon, preprocess_images_toon_test
from tf_slim.utils import get_variables_to_train
from utils import montage

slim = tf.contrib.slim

# Setup training parameters
data = cifar10
TRAIN_SET_NAME = 'train'
model = AEGAN2(num_layers=5, batch_size=128, data_size=data.SPLITS_TO_SIZES[TRAIN_SET_NAME], num_epochs=100)
train_ae = False
TEST_SET_NAME = 'test'
TARGET_SHAPE = [64, 64, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)
SAVE_DIR = os.path.join(LOG_DIR, '{}_{}/'.format(data.NAME, model.name))

tf.logging.set_verbosity(tf.logging.DEBUG)

sess = tf.Session()
g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Get the training dataset
        train_set = data.get_split(TRAIN_SET_NAME)
        provider = slim.dataset_data_provider.DatasetDataProvider(train_set,
                                                                  num_readers=8,
                                                                  common_queue_capacity=32 * model.batch_size,
                                                                  common_queue_min=8 * model.batch_size)
        [img_train, edge_train, toon_train] = provider.get(['image', 'edges', 'cartoon'])

        # Get some test-data
        test_set = data.get_split(TEST_SET_NAME)
        provider = slim.dataset_data_provider.DatasetDataProvider(test_set, shuffle=False)
        [img_test, edge_test, toon_test] = provider.get(['image', 'edges', 'cartoon'])

        # Pre-process training data
        with tf.device('/cpu:0'):
            img_train, edge_train, toon_train = preprocess_images_toon(
                img_train, edge_train, toon_train,
                output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1],
                resize_side_min=RESIZE_SIZE,
                resize_side_max=int(RESIZE_SIZE * 1.5))
            img_test, edge_test, toon_test = preprocess_images_toon_test(
                img_test, edge_test, toon_test,
                output_height=TARGET_SHAPE[0], output_width=TARGET_SHAPE[1],
                resize_side=RESIZE_SIZE)

        # Make batches
        imgs_train, edges_train, toons_train = tf.train.batch([img_train, edge_train, toon_train],
                                                              batch_size=model.batch_size, num_threads=8,
                                                              capacity=8 * model.batch_size)
        imgs_test, edges_test, toons_test = tf.train.batch([img_train, edge_train, toon_train],
                                                           batch_size=model.batch_size)

        # Get labels for discriminator training
        labels_disc = model.disc_labels()
        labels_gen = model.gen_labels()
        if train_ae:
            labels_ae = model.ae_labels()

        # Create the model
        img_rec, gen_rec, disc_out, enc_im, gen_enc = model.net(imgs_train, toons_train, edges_train)
        img_rec_test, gen_rec_test, _, _, _ = model.net(imgs_test, toons_test, edges_test, reuse=True, training=False)

        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        dL_disc = slim.losses.softmax_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope, weight=1.0)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Define the losses for AE training
        ae_loss_scope = 'ae_loss'
        if train_ae:
            dL_ae = slim.losses.softmax_cross_entropy(disc_out, labels_ae, scope=ae_loss_scope, weight=1.0)
        l2_ae = slim.losses.sum_of_squares(img_rec, imgs_train, scope=ae_loss_scope, weight=100.0)
        losses_ae = slim.losses.get_losses(ae_loss_scope)
        losses_ae += slim.losses.get_regularization_losses(ae_loss_scope)
        ae_loss = math_ops.add_n(losses_ae, name='ae_total_loss')

        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        dL_gen = slim.losses.softmax_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope, weight=1.0)
        l2_gen = slim.losses.sum_of_squares(gen_rec, imgs_train, scope=gen_loss_scope, weight=50)
        for lg, le in zip(gen_enc, enc_im):
            slim.losses.sum_of_squares(lg, le, scope=gen_loss_scope, weight=10.0)
        losses_gen = slim.losses.get_losses(gen_loss_scope)
        losses_gen += slim.losses.get_regularization_losses(gen_loss_scope)
        gen_loss = math_ops.add_n(losses_gen, name='gen_total_loss')

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        # Handle dependencies with update_ops (batch-norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            gen_loss = control_flow_ops.with_dependencies([updates], gen_loss)
            ae_loss = control_flow_ops.with_dependencies([updates], ae_loss)
            disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

        # Define learning rate
        decay_steps = int(data.SPLITS_TO_SIZES[TRAIN_SET_NAME] / model.batch_size)
        learning_rate = tf.train.exponential_decay(0.001,
                                                   global_step,
                                                   decay_steps,
                                                   0.97,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Handle summaries
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.scalar_summary('losses/disc-loss generator', dL_gen)
        if train_ae:
            tf.scalar_summary('losses/disc-loss ae', dL_ae)
        tf.scalar_summary('losses/l2 generator', l2_gen)
        tf.scalar_summary('losses/l2 auto-encoder', l2_ae)
        tf.scalar_summary('learning rate', learning_rate)
        tf.image_summary('images/generator', montage(gen_rec_test, 8, 8), max_images=1)
        tf.image_summary('images/ae', montage(img_rec_test, 8, 8), max_images=1)
        tf.image_summary('images/ground-truth', montage(imgs_test, 8, 8), max_images=1)
        tf.image_summary('images/cartoons', montage(toons_test, 8, 8), max_images=1)
        tf.image_summary('images/edges', montage(edges_test, 8, 8), max_images=1)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, epsilon=1e-6)

        # Generator training operation
        scopes_gen = 'generator'
        vars2train_gen = get_variables_to_train(trainable_scopes=scopes_gen)
        train_op_gen = slim.learning.create_train_op(gen_loss, optimizer, variables_to_train=vars2train_gen,
                                                     global_step=global_step, summarize_gradients=False)

        # Auto-encoder training operation
        scopes_ae = 'encoder, decoder'
        vars2train_ae = get_variables_to_train(trainable_scopes=scopes_ae)
        train_op_ae = slim.learning.create_train_op(ae_loss, optimizer, variables_to_train=vars2train_ae,
                                                    global_step=global_step, summarize_gradients=False)

        # Discriminator training operation
        scopes_disc = 'discriminator'
        vars2train_disc = get_variables_to_train(trainable_scopes=scopes_disc)
        train_op_disc = slim.learning.create_train_op(disc_loss, optimizer, variables_to_train=vars2train_disc,
                                                      global_step=global_step, summarize_gradients=False)

        # Start training
        num_train_steps = (data.SPLITS_TO_SIZES[TRAIN_SET_NAME] / model.batch_size) * model.num_ep
        slim.learning.train(train_op_ae + train_op_gen + train_op_disc,
                            SAVE_DIR,
                            save_summaries_secs=300,
                            save_interval_secs=3000,
                            log_every_n_steps=100,
                            number_of_steps=num_train_steps)
