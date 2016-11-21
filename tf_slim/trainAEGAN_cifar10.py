import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNet import AEGAN2
from datasets import cifar10
from preprocess import preprocess_images_toon
from tf_slim.utils import get_variables_to_train
from utils import montage

slim = tf.contrib.slim

# Setup training parameters
NUM_LAYERS = 4
BATCH_SIZE = 128
TARGET_SHAPE = [32, 32, 3]
NUM_EPOCHS = 50
LOG_DIR = '/data/cvg/simon/data/logs/cifar10_aegan2/'
SET_NAME = 'train'
data = cifar10

tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.Session()
g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Get the dataset
        dataset = data.get_split(SET_NAME)
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  num_readers=8,
                                                                  common_queue_capacity=32 * BATCH_SIZE,
                                                                  common_queue_min=8 * BATCH_SIZE)
        [image, edge, cartoon] = provider.get(['image', 'edges', 'cartoon'])

        # Pre-process training data
        with tf.device('/cpu:0'):
            image, edge, cartoon = preprocess_images_toon(image, edge, cartoon,
                                                          output_height=TARGET_SHAPE[0],
                                                          output_width=TARGET_SHAPE[1],
                                                          resize_side_min=data.MIN_SIZE,
                                                          resize_side_max=int(data.MIN_SIZE * 1.5))

        # Make batches
        images, edges, cartoons = tf.train.batch([image, edge, cartoon],
                                                 batch_size=BATCH_SIZE,
                                                 num_threads=8,
                                                 capacity=8 * BATCH_SIZE)

        # Make labels for discriminator training
        labels = tf.Variable(tf.concat(concat_dim=0, values=[tf.zeros(shape=(BATCH_SIZE/2,), dtype=tf.int32),
                                                             tf.ones(shape=(BATCH_SIZE/2,), dtype=tf.int32)]))
        labels = tf.random_shuffle(labels)
        labels_disc = slim.one_hot_encoding(labels, 2)
        labels_gen = slim.one_hot_encoding(tf.ones_like(labels) - labels, 2)

        # Create the model
        img_rec, gen_rec, disc_out, enc_im, gen_enc = AEGAN2(images, cartoons, edges,
                                                            num_layers=NUM_LAYERS,
                                                            order=labels)

        # Define loss for discriminator training
        disc_loss_scope = 'disc_loss'
        dL_disc = slim.losses.sigmoid_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope, weight=0.5)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Define the losses for AE training
        ae_loss_scope = 'ae_loss'
        dL_ae = slim.losses.sigmoid_cross_entropy(disc_out, labels_disc, scope=ae_loss_scope, weight=0.5)
        l2_ae = slim.losses.sum_of_squares(img_rec, images, scope=ae_loss_scope, weight=25.0)
        losses_ae = slim.losses.get_losses(ae_loss_scope)
        losses_ae += slim.losses.get_regularization_losses(ae_loss_scope)
        ae_loss = math_ops.add_n(losses_ae, name='ae_total_loss')

        # Define the losses for generator training
        gen_loss_scope = 'gen_loss'
        dL_gen = slim.losses.sigmoid_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope, weight=1.0)
        l2_gen = slim.losses.sum_of_squares(gen_rec, images, scope=gen_loss_scope, weight=5.0)
        l2feat_gen = slim.losses.sum_of_squares(gen_enc, enc_im, scope=gen_loss_scope, weight=1.0)
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
        decay_steps = int(data.SPLITS_TO_SIZES[SET_NAME] / BATCH_SIZE)
        learning_rate = tf.train.exponential_decay(0.005,
                                                   global_step,
                                                   decay_steps,
                                                   0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Handle summaries
        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.scalar_summary('losses/disc-loss generator', dL_gen)
        tf.scalar_summary('losses/disc-loss auto-encoder', dL_ae)
        tf.scalar_summary('losses/l2 generator', l2_gen)
        tf.scalar_summary('losses/l2feat generator', l2feat_gen)
        tf.scalar_summary('losses/l2 auto-encoder', l2_ae)
        tf.scalar_summary('learning rate', learning_rate)
        tf.image_summary('images/generator', montage(gen_rec, 8, 8), max_images=1)
        tf.image_summary('images/ae', montage(img_rec, 8, 8), max_images=1)
        tf.image_summary('images/ground-truth', montage(images, 8, 8), max_images=1)
        tf.image_summary('images/cartoons', montage(cartoons, 8, 8), max_images=1)
        tf.image_summary('images/edges', montage(edges, 8, 8), max_images=1)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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
        num_train_steps = (data.SPLITS_TO_SIZES[SET_NAME] / BATCH_SIZE) * NUM_EPOCHS
        slim.learning.train(train_op_ae + train_op_gen + train_op_disc,
                            LOG_DIR,
                            save_summaries_secs=120,
                            save_interval_secs=3000,
                            log_every_n_steps=100,
                            number_of_steps=num_train_steps)
