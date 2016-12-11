import os

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNetMaxPool import AEGAN
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_toon_train, preprocess_toon_test
from tf_slim.utils import get_variables_to_train
from utils import montage, assign_from_checkpoint_fn

slim = tf.contrib.slim

# Setup training parameters
data = cifar10
TRAIN_SET_NAME = 'train'
TEST_SET_NAME = 'test'
model = AEGAN(num_layers=4, batch_size=512, data_size=data.SPLITS_TO_SIZES[TRAIN_SET_NAME], num_epochs=300)
TARGET_SHAPE = [32, 32, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)
SAVE_DIR = os.path.join(LOG_DIR, '{}_{}_GAN/'.format(data.NAME, model.name))
CHECKPOINT = 'model.ckpt-29100'
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_ae_gen/{}'.format(data.NAME, model.name, CHECKPOINT))
TEST = False

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
        gen_rec, disc_out, enc_im, gen_enc = model.gan(imgs_train, toons_train, edges_train)

        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        dL_disc = slim.losses.softmax_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope, weight=1.0)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        dL_gen = slim.losses.softmax_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope, weight=1.0)
        l2_gen = slim.losses.sum_of_squares(gen_rec, imgs_train, scope=gen_loss_scope, weight=50)
        l2_feat = slim.losses.sum_of_squares(gen_enc, enc_im, scope=gen_loss_scope, weight=10.0)
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
            disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

        # Define learning parameters
        num_train_steps = (data.SPLITS_TO_SIZES[TRAIN_SET_NAME] / model.batch_size) * model.num_ep
        learning_rate = tf.select(tf.python.math_ops.greater(global_step, num_train_steps / 2),
                                  0.0002 - 0.0002 * (2*tf.cast(global_step, tf.float32)/num_train_steps-1.0), 0.0002)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

        # Handle summaries
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.scalar_summary('losses/disc-loss generator', dL_gen)
        tf.scalar_summary('losses/l2 generator', l2_gen)
        tf.scalar_summary('losses/l2 features', l2_feat)

        if TEST:
            gen_rec_test, _, _, _ = model.gan(imgs_test, toons_test, edges_test, reuse=True, training=False)
            tf.image_summary('images/generator', montage(gen_rec_test, 8, 8), max_images=1)
            tf.image_summary('images/ground-truth', montage(imgs_test, 8, 8), max_images=1)
            tf.image_summary('images/cartoons', montage(toons_test, 8, 8), max_images=1)
            tf.image_summary('images/edges', montage(edges_test, 8, 8), max_images=1)
        else:
            tf.image_summary('images/generator', montage(gen_rec, 8, 8), max_images=1)
            tf.image_summary('images/ground-truth', montage(imgs_train, 8, 8), max_images=1)
            tf.image_summary('images/cartoons', montage(toons_train, 8, 8), max_images=1)
            tf.image_summary('images/edges', montage(edges_train, 8, 8), max_images=1)

        # Generator training operation
        scopes_gen = 'generator'
        vars2train_gen = get_variables_to_train(trainable_scopes=scopes_gen)
        train_op_gen = slim.learning.create_train_op(gen_loss, optimizer, variables_to_train=vars2train_gen,
                                                     global_step=global_step, summarize_gradients=False)

        # Discriminator training operation
        scopes_disc = 'discriminator'
        vars2train_disc = get_variables_to_train(trainable_scopes=scopes_disc)
        train_op_disc = slim.learning.create_train_op(disc_loss, optimizer, variables_to_train=vars2train_disc,
                                                      global_step=global_step, summarize_gradients=False)

        # Specify the layers of your model you want to exclude
        variables_to_restore = slim.get_variables_to_restore(include=['decoder', 'encoder', 'generator'])
        print('Variables to restore: {}'.format([v.op.name for v in variables_to_restore]))
        init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        slim.learning.train(train_op_gen + train_op_disc,
                            SAVE_DIR,
                            init_fn=init_fn,
                            save_summaries_secs=300,
                            save_interval_secs=3000,
                            log_every_n_steps=100,
                            number_of_steps=num_train_steps)