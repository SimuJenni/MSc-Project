import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNetAEGAN import VAEGAN
from constants import LOG_DIR
from datasets import stl10
from preprocess import preprocess_toon_train, preprocess_toon_test
from tf_slim.utils import get_variables_to_train
from utils import montage

slim = tf.contrib.slim

# Setup training parameters
data = stl10
TRAIN_SET_NAME = 'train_unlabeled'
TEST_SET_NAME = 'test'
model = VAEGAN(num_layers=5, batch_size=64, data_size=data.SPLITS_TO_SIZES[TRAIN_SET_NAME], num_epochs=150)
TARGET_SHAPE = [96, 96, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)
SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_new/'.format(data.NAME, model.name))
TEST = False
NUM_IMG_SUMMARY = 6

tf.logging.set_verbosity(tf.logging.DEBUG)
sess = tf.Session()
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
            [img_train, edge_train, toon_train] = provider.get(['image', 'edges', 'cartoon'])

            # Preprocess data
            img_train, edge_train, toon_train = preprocess_toon_train(img_train, edge_train, toon_train,
                                                                      output_height=TARGET_SHAPE[0],
                                                                      output_width=TARGET_SHAPE[1],
                                                                      resize_side_min=RESIZE_SIZE,
                                                                      resize_side_max=int(RESIZE_SIZE * 1.5))
            # Make batches
            imgs_train, edges_train, toons_train = tf.train.batch([img_train, edge_train, toon_train],
                                                                  batch_size=model.batch_size, num_threads=8,
                                                                  capacity=4 * model.batch_size)
            if TEST:
                # Get test-data
                test_set = data.get_split('test')
                provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
                [img_test, edge_test, toon_test] = provider.get(['image', 'edges', 'cartoon'])
                img_test, edge_test, toon_test = preprocess_toon_test(img_test, edge_test, toon_test,
                                                                      output_height=TARGET_SHAPE[0],
                                                                      output_width=TARGET_SHAPE[1],
                                                                      resize_side=RESIZE_SIZE)
                imgs_test, edges_test, toons_test = tf.train.batch([img_test, edge_test, toon_test],
                                                                   batch_size=model.batch_size, num_threads=4)

        # Get labels for discriminator training
        labels_disc = model.disc_labels()
        labels_gen = model.gen_labels()

        # Create the model
        img_rec, gen_rec, disc_out, enc_dist, gen_dist, enc_mu, gen_mu, enc_logvar, gen_logvar = \
            model.net(imgs_train, toons_train, edges_train)

        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        dL_disc = slim.losses.softmax_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope, weight=1.0)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Define the losses for AE training
        ae_loss_scope = 'ae_loss'
        l2_ae = slim.losses.sum_of_squares(img_rec, imgs_train, scope=ae_loss_scope, weight=100)
        # kl_ae = kl_gauss(z_log_sigma=enc_logvar/2.0, z_mean=enc_mu)
        losses_ae = slim.losses.get_losses(ae_loss_scope)
        losses_ae += slim.losses.get_regularization_losses(ae_loss_scope)
        ae_loss = math_ops.add_n(losses_ae, name='ae_total_loss')

        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        dL_gen = slim.losses.softmax_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope, weight=1.0)
        l2_gen = slim.losses.sum_of_squares(gen_rec, imgs_train, scope=gen_loss_scope, weight=30.0)
        l2_mu = slim.losses.sum_of_squares(gen_mu, enc_mu, scope=gen_loss_scope, weight=3.0)
        l2_sigma = slim.losses.sum_of_squares(gen_logvar, enc_logvar, scope=gen_loss_scope, weight=3.0)
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

        # Define learning parameters
        num_train_steps = (data.SPLITS_TO_SIZES[TRAIN_SET_NAME] / model.batch_size) * model.num_ep
        learning_rate = tf.select(tf.python.math_ops.greater(global_step, num_train_steps / 2),
                                  0.0002 - 0.0002 * (2*tf.cast(global_step, tf.float32)/num_train_steps-1.0), 0.0002)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)

        # Handle summaries
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.scalar_summary('losses/disc-loss generator', dL_gen)
        tf.scalar_summary('losses/l2 generator', l2_gen)
        tf.scalar_summary('losses/L2 mu', l2_mu)
        tf.scalar_summary('losses/L2 sigma', l2_sigma)
        tf.scalar_summary('losses/l2 auto-encoder', l2_ae)
        # tf.scalar_summary('losses/kl auto-encoder', kl_ae)

        if TEST:
            img_rec_test, gen_rec_test, _, _, _ = model.net(imgs_test, toons_test, edges_test, reuse=True,
                                                            training=False)
            tf.image_summary('images/generator', montage(gen_rec_test, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/ae', montage(img_rec_test, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/ground-truth', montage(imgs_test, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/cartoons', montage(toons_test, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/edges', montage(edges_test, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
        else:
            tf.image_summary('images/generator', montage(gen_rec, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/ae', montage(img_rec, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/ground-truth', montage(imgs_train, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/cartoons', montage(toons_train, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)
            tf.image_summary('images/edges', montage(edges_train, NUM_IMG_SUMMARY, NUM_IMG_SUMMARY), max_images=1)

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
        slim.learning.train(train_op_ae + train_op_gen + train_op_disc,
                            SAVE_DIR,
                            save_summaries_secs=300,
                            save_interval_secs=3000,
                            log_every_n_steps=100,
                            number_of_steps=num_train_steps)
