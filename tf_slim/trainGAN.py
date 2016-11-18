import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from ToonNet import GANAE
from datasets import imagenet
from preprocess import preprocess_image
from tf_slim.utils import get_variables_to_train

slim = tf.contrib.slim

NUM_LAYERS = 3
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 64
NUM_CLASSES = 1000
IM_SHAPE = [224, 224, 3]

MODEL_PATH = '/data/cvg/qhu/try_GAN/checkpoint_edge_advplus_128/010/DCGAN.model-148100'
LOG_DIR = '/data/cvg/simon/data/logs/alex_net_v2_2/'

tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.Session()
g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        dataset = imagenet.get_split('train', DATA_DIR)
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                                  num_readers=8,
                                                                  common_queue_capacity=32 * BATCH_SIZE,
                                                                  common_queue_min=8 * BATCH_SIZE)

        [image, edge, cartoon] = provider.get(['image', 'edges', 'cartoon'])

        image = preprocess_image(image,
                                 is_training=True,
                                 output_height=IM_SHAPE[0],
                                 output_width=IM_SHAPE[1])

        images, edges, cartoons = tf.train.batch([image, edge, cartoon],
                                                 batch_size=BATCH_SIZE,
                                                 num_threads=8,
                                                 capacity=8 * BATCH_SIZE)

        order = tf.placeholder(tf.int16, shape=(BATCH_SIZE,))

        # Create the model
        img_rec, gen_rec, disc_out = GANAE(images, cartoons, edges, order, num_layers=NUM_LAYERS)

        # Define the losses for discriminator training
        disc_loss_scope = 'disc_loss'
        labels_disc = slim.one_hot_encoding(order, 2)
        slim.losses.sigmoid_cross_entropy(disc_out, labels_disc, scope=disc_loss_scope)
        slim.losses.sum_of_squares(img_rec, images, scope=disc_loss_scope)
        losses_disc = slim.losses.get_losses(disc_loss_scope)
        losses_disc += slim.losses.get_regularization_losses(disc_loss_scope)
        disc_loss = math_ops.add_n(losses_disc, name='disc_total_loss')

        # Define the losses for discriminator training
        gen_loss_scope = 'gen_loss'
        labels_gen = slim.one_hot_encoding(tf.ones_like(order) - order, 2)
        slim.losses.sigmoid_cross_entropy(disc_out, labels_gen, scope=gen_loss_scope)
        slim.losses.sum_of_squares(gen_rec, images, scope=gen_loss_scope)
        losses_gen = slim.losses.get_losses(gen_loss_scope)
        losses_gen += slim.losses.get_regularization_losses(gen_loss_scope)
        gen_loss = math_ops.add_n(losses_disc, name='gen_total_loss')

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            gen_loss = control_flow_ops.with_dependencies([updates], gen_loss)
            disc_loss = control_flow_ops.with_dependencies([updates], disc_loss)

        tf.scalar_summary('losses/discriminator loss', disc_loss)
        tf.scalar_summary('losses/generator loss', gen_loss)

        decay_steps = int(imagenet._SPLITS_TO_SIZES['train'] / BATCH_SIZE * 2.0)
        learning_rate = tf.train.exponential_decay(0.01,
                                                   global_step,
                                                   decay_steps,
                                                   0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1.0, momentum=0.9, decay=0.9)

        # Create training operations
        scopes_gen = ['generator']
        vars2train_gen = get_variables_to_train(trainable_scopes=scopes_gen)
        train_op_gen = slim.learning.create_train_op(gen_loss, optimizer, variables_to_train=vars2train_gen,
                                                     global_step=global_step, summarize_gradients=True)

        scopes_disc = ['encoder', 'decoder', 'discriminator']
        vars2train_disc = get_variables_to_train(trainable_scopes=scopes_disc)
        train_op_disc = slim.learning.create_train_op(disc_loss, optimizer, variables_to_train=vars2train_disc,
                                                      global_step=global_step, summarize_gradients=True)

        # Start training.
        slim.learning.train(train_op_disc + train_op_gen, LOG_DIR, save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
