import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import sys
import numpy as np

from utils import montage_tf, get_variables_to_train, assign_from_checkpoint_fn
from constants import LOG_DIR

slim = tf.contrib.slim


class ToonNet_Trainer:
    def __init__(self, model, dataset, pre_processor, num_epochs, optimizer='adam', lr_policy='const', init_lr=0.0002,
                 tag='default'):
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.save_dir = os.path.join(LOG_DIR, '{}_{}_{}/'.format(dataset.name, model.name, tag))
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.opt_type = optimizer
        self.lr_policy = lr_policy
        self.init_lr = init_lr
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def optimizer(self):
        opts = {'adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate(), beta1=0.5, epsilon=1e-6),
                'sgd+momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate(), momentum=0.9)}
        return opts[self.opt_type]

    def learning_rate(self):
        policies = {'const': self.init_lr,
                    'step': self.learning_rate_alex(),
                    'linear': self.learning_rate_linear(self.init_lr)}
        return policies[self.lr_policy]

    def get_toon_train_batch(self):
        with tf.device('/cpu:0'):
            # Get the training dataset
            train_set = self.dataset.get_toon_train()
            provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=4,
                                                                      common_queue_capacity=2 * self.model.batch_size,
                                                                      common_queue_min=self.model.batch_size)
            [img_train, edge_train, toon_train] = provider.get(['image', 'edges', 'cartoon'])

            # Preprocess data
            img_train, edge_train, toon_train = self.pre_processor.process_train_toonnet(img_train, edge_train,
                                                                                         toon_train)
            # Make batches
            imgs_train, edges_train, toons_train = tf.train.batch([img_train, edge_train, toon_train],
                                                                  batch_size=self.model.batch_size,
                                                                  num_threads=8,
                                                                  capacity=self.model.batch_size)
        return imgs_train, edges_train, toons_train

    def get_finetune_batch(self):
        # Get the training dataset
        train_set = self.dataset.get_trainset()
        provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=2,
                                                                  common_queue_capacity=4 * self.model.batch_size,
                                                                  common_queue_min=self.model.batch_size)
        images_and_labels = []
        for thread_id in range(4):
            # Parse a serialized Example proto to extract the image and metadata.
            [img_train, label_train] = provider.get(['image', 'label'])
            label_train -= self.dataset.label_offset

            # Pre-process data
            img_train = self.pre_processor.process_transfer_train(img_train, thread_id)
            images_and_labels.append([img_train, label_train])

        # Make batches
        imgs_train, labels_train = tf.train.batch_join(
            images_and_labels,
            batch_size=self.model.batch_size,
            capacity=4 * self.model.batch_size)

        return imgs_train, labels_train

    def classification_loss(self, preds_train, labels_train):
        # Define the loss
        loss_scope = 'classification_loss'
        train_loss = slim.losses.softmax_cross_entropy(preds_train,
                                                       labels_train,
                                                       scope=loss_scope)
        tf.scalar_summary('losses/training loss', train_loss)
        train_losses = slim.losses.get_losses(loss_scope)
        train_losses += slim.losses.get_regularization_losses(loss_scope)
        total_train_loss = math_ops.add_n(train_losses, name='total_train_loss')

        # Compute accuracy
        predictions = tf.argmax(preds_train, 1)
        tf.scalar_summary('accuracy/training accuracy', slim.metrics.accuracy(predictions, labels_train))
        return total_train_loss

    def discriminator_loss(self, disc_out, disc_labels):
        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        disc_loss = slim.losses.softmax_cross_entropy(disc_out, disc_labels, scope=disc_loss_scope, weight=1.0)
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_total_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Compute accuracy
        predictions = tf.argmax(disc_out, 1)
        tf.scalar_summary('accuracy/discriminator accuracy', slim.metrics.accuracy(predictions, tf.argmax(disc_labels, 1)))
        return disc_total_loss

    def autoencoder_loss(self, imgs_rec, imgs_train):
        # Define the losses for AE training
        ae_loss_scope = 'ae_loss'
        ae_loss = slim.losses.sum_of_squares(imgs_rec, imgs_train, scope=ae_loss_scope, weight=30)
        tf.scalar_summary('losses/autoencoder loss (encoder+decoder)', ae_loss)
        losses_ae = slim.losses.get_losses(ae_loss_scope)
        losses_ae += slim.losses.get_regularization_losses(ae_loss_scope)
        ae_total_loss = math_ops.add_n(losses_ae, name='ae_total_loss')
        return ae_total_loss

    def generator_loss(self, disc_out, labels_gen, imgs_gen, imgs_train, g_mu, g_var, e_mu, e_var):
        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        gen_disc_loss = slim.losses.softmax_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope, weight=1.0)
        tf.scalar_summary('losses/discriminator loss (generator)', gen_disc_loss)
        gen_ae_loss = slim.losses.sum_of_squares(imgs_gen, imgs_train, scope=gen_loss_scope, weight=30.0)
        tf.scalar_summary('losses/autoencoder loss (generator)', gen_ae_loss)
        gen_mu_loss = slim.losses.sum_of_squares(g_mu, e_mu, scope=gen_loss_scope, weight=3.0)
        tf.scalar_summary('losses/mu loss (generator)', gen_mu_loss)
        gen_var_loss = slim.losses.sum_of_squares(g_var, e_var, scope=gen_loss_scope, weight=3.0)
        tf.scalar_summary('losses/var loss (generator)', gen_var_loss)
        losses_gen = slim.losses.get_losses(gen_loss_scope)
        losses_gen += slim.losses.get_regularization_losses(gen_loss_scope)
        gen_loss = math_ops.add_n(losses_gen, name='gen_total_loss')
        return gen_loss

    def make_train_op(self, loss, vars2train=None, scope=None):
        if scope:
            vars2train = get_variables_to_train(trainable_scopes=scope)
        train_op = slim.learning.create_train_op(loss, self.optimizer(), variables_to_train=vars2train,
                                                 global_step=self.global_step, summarize_gradients=False)
        return train_op

    def num_train_steps(self):
        return (self.dataset.num_train / self.model.batch_size) * self.num_epochs

    def make_summaries(self):
        # Handle summaries
        for variable in slim.get_model_variables():
            tf.histogram_summary(variable.op.name, variable)
        tf.scalar_summary('learning rate', self.learning_rate())

    def make_image_summaries(self, edges_train, img_gen, img_rec, imgs_train, toons_train):
        tf.image_summary('imgs/generator out', montage_tf(img_gen, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/autoencoder', montage_tf(img_rec, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/ground truth', montage_tf(imgs_train, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/cartoons', montage_tf(toons_train, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/edge maps', montage_tf(edges_train, 1, self.im_per_smry), max_images=1)

    def learning_rate_alex(self):
        # Define learning rate schedule
        num_train_steps = self.num_train_steps()
        boundaries = [np.int64(num_train_steps * 0.2), np.int64(num_train_steps * 0.4),
                      np.int64(num_train_steps * 0.6), np.int64(num_train_steps * 0.8)]
        values = [0.01, 0.01 * 250. ** (-1. / 4.), 0.01 * 250 ** (-2. / 4.), 0.01 * 250 ** (-3. / 4.),
                  0.01 * 250. ** (-1.)]
        return tf.train.piecewise_constant(self.global_step, boundaries=boundaries, values=values)

    def learning_rate_linear(self, init_lr=0.0002):
        return tf.train.polynomial_decay(init_lr, self.global_step, self.num_train_steps(), end_learning_rate=0.0)

    def get_variables_to_transfer(self, num_conv):
        var2train = []
        for i in range(num_conv):
            vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(self.model.num_layers - i)],
                                               exclude=['discriminator/fully_connected'])
            vs = list(set(vs).intersection(tf.trainable_variables()))
            var2train += vs
        vs = slim.get_variables_to_restore(include=['fully_connected'],
                                           exclude=['discriminator/fully_connected'])
        vs = list(set(vs).intersection(tf.trainable_variables()))
        var2train += vs
        print('Variables to train: {}'.format([v.op.name for v in var2train]))
        return var2train

    def make_init_fn(self, chpt_path, num_conv2init):
        # Specify the layers of the model you want to exclude
        var2restore = []
        for i in range(num_conv2init):
            vs = slim.get_variables_to_restore(include=['discriminator/conv_{}'.format(i + 1)],
                                               exclude=['discriminator/fully_connected'])
            var2restore += vs
        init_fn = assign_from_checkpoint_fn(chpt_path, var2restore, ignore_missing_vars=True)
        print('Variables to restore: {}'.format([v.op.name for v in var2restore]))
        return init_fn

    def train(self):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        sess = tf.Session()
        g = tf.Graph()
        with sess.as_default():
            with g.as_default():
                imgs_train, edges_train, toons_train = self.get_toon_train_batch()

                # Get labels for discriminator training
                labels_disc = self.model.disc_labels()
                labels_gen = self.model.gen_labels()

                # Create the model
                img_rec, img_gen, disc_out, e_mu, g_mu, e_var, g_var = \
                    self.model.net(imgs_train, toons_train, edges_train)

                # Compute losses
                disc_loss = self.discriminator_loss(disc_out, labels_disc)
                ae_loss = self.autoencoder_loss(img_rec, imgs_train)
                gen_loss = self.generator_loss(disc_out, labels_gen, img_gen, imgs_train, g_mu, g_var, e_mu, e_var)

                # Handle dependencies with update_ops (batch-norm)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    gen_loss = control_flow_ops.with_dependencies([updates], gen_loss)
                    ae_loss = control_flow_ops.with_dependencies([updates], ae_loss)
                    disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

                # Make summaries
                self.make_summaries()
                self.make_image_summaries(edges_train, img_gen, img_rec, imgs_train, toons_train)

                # Generator training operations
                train_op_gen = self.make_train_op(gen_loss, scope='generator')
                train_op_ae = self.make_train_op(ae_loss, scope='encoder, decoder')
                train_op_disc = self.make_train_op(disc_loss, scope='discriminator')

                # Start training
                slim.learning.train(train_op_ae + train_op_gen + train_op_disc, self.save_dir,
                                    save_summaries_secs=600,
                                    save_interval_secs=3000,
                                    log_every_n_steps=100,
                                    number_of_steps=self.num_train_steps())

    def transfer_finetune(self, chpt_path, num_conv2train=None, num_conv2init=None):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_train, labels_train = self.get_finetune_batch()

                # Get predictions
                preds_train = self.model.build_classifier(imgs_train, self.dataset.num_classes)

                # Compute the loss
                total_train_loss = self.classification_loss(
                    preds_train, self.dataset.format_labels(labels_train))

                # Handle dependencies
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    total_train_loss = control_flow_ops.with_dependencies([updates], total_train_loss)

                # Create training operation
                if num_conv2train:
                    var2train = self.get_variables_to_transfer(num_conv2train)
                else:
                    var2train = tf.trainable_variables()
                train_op = self.make_train_op(total_train_loss, vars2train=var2train)

                # Start training
                sys.stdout.flush()
                slim.learning.train(train_op, self.save_dir,
                                    init_fn=self.make_init_fn(chpt_path, num_conv2init),
                                    number_of_steps=self.num_train_steps(),
                                    save_summaries_secs=300, save_interval_secs=600,
                                    log_every_n_steps=100)

