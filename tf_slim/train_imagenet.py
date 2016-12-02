from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

from alexnet import alexnet_v2, alexnet_v2_arg_scope
from datasets import imagenet
from preprocess import preprocess_image
from qhu_code.model_edge_advplus_128 import DCGAN
from utils import get_variables_to_train, assign_from_checkpoint_fn

slim = tf.contrib.slim

FINE_TUNE = False
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 128
NUM_CLASSES = 1000
NUM_EP = 100
IM_SHAPE = [224, 224, 3]
PRE_TRAINED_SCOPE = 'pre_trained_scope'
MODEL_PATH = '/data/cvg/qhu/try_GAN/checkpoint_edge_advplus_128/010/DCGAN.model-148100'
LOG_DIR = '/data/cvg/?'  # TODO: specify log-dir

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)


def Classifier(inputs, fine_tune=False, training=True, reuse=None):
    """Build a classifier on the given inputs. If fine_tune=False it builds an AlexNet.
    Args:
        inputs: Placeholder of image-batch
        fine_tune: Whether to build classifier on pretrained model or train AlexNet
        training: Whether in train or test mode
        reuse: Whether to reuse already existing variables
    Returns:
        Logit outputs from the classifier
    """
    if fine_tune:
        # Build the model to fine-tune
        model = DCGAN(sess, batch_size=BATCH_SIZE, is_train=training, image_shape=IM_SHAPE)
        with tf.variable_scope(PRE_TRAINED_SCOPE, reuse=reuse):
            # TODO: Specify model to fine-tune here
            net = model.discriminator(inputs)

        # Build classifier on top
        batch_norm_params = {'decay': 0.999, 'epsilon': 0.001}
        with tf.variable_scope('fully_connected', reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                net = slim.flatten(net)
                net = slim.fully_connected(net, 4096, scope='fc1')
                net = slim.dropout(net)
                net = slim.fully_connected(net, 4096, scope='fc2')
                net = slim.dropout(net)
                net = slim.fully_connected(net, NUM_CLASSES, scope='fc3',
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           biases_initializer=tf.zeros_initializer)
        return net
    else:
        # Train AlexNet
        with slim.arg_scope(alexnet_v2_arg_scope()):
            net = alexnet_v2(inputs, is_training=training, reuse=reuse)
        return net


g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Pre-process training data
        with tf.device('/cpu:0'):
            # Get the training dataset
            dataset = imagenet.get_split('train', DATA_DIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=8,
                                                                      common_queue_capacity=32 * BATCH_SIZE,
                                                                      common_queue_min=8 * BATCH_SIZE)
            [img_train, label] = provider.get(['image', 'label'])

            # Get some test-data
            test_set = imagenet.get_split('validation', DATA_DIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
            [img_test, label_test] = provider.get(['image', 'label'])

            img_train = preprocess_image(img_train, is_training=True, output_height=IM_SHAPE[0],
                                         output_width=IM_SHAPE[1])
            img_test = preprocess_image(img_test, is_training=False, output_height=IM_SHAPE[0],
                                        output_width=IM_SHAPE[1])
            if FINE_TUNE:
                img_train = tf.to_float(img_train) * (2. / 255.)
                img_test = tf.to_float(img_test) * (2. / 255.)

            # Make batches
            imgs_train, labels_train = tf.train.batch([img_train, label], batch_size=BATCH_SIZE, num_threads=8,
                                                      capacity=8 * BATCH_SIZE)
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=BATCH_SIZE, num_threads=4)

        # Create the model
        predictions = Classifier(imgs_train, FINE_TUNE)
        preds_test = Classifier(imgs_test, FINE_TUNE, training=False, reuse=True)

        # Define the loss
        train_loss = slim.losses.softmax_cross_entropy(predictions, slim.one_hot_encoding(labels_train, NUM_CLASSES))
        total_loss = slim.losses.get_total_loss()
        test_loss = slim.losses.softmax_cross_entropy(preds_test, slim.one_hot_encoding(labels_test, NUM_CLASSES))

        # Handle dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates], total_loss)

        # Compute predictions for accuracy computation
        preds_train = tf.argmax(predictions, 1)
        preds_test = tf.argmax(preds_test, 1)

        # Define learning rate
        num_train_steps = (imagenet.SPLITS_TO_SIZES['train'] / BATCH_SIZE) * NUM_EP
        learning_rate = tf.train.exponential_decay(0.01,
                                                   global_step,
                                                   (imagenet.SPLITS_TO_SIZES['train'] / BATCH_SIZE),
                                                   0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.001)

        # Gather all summaries.
        tf.scalar_summary('learning rate', learning_rate)
        tf.scalar_summary('losses/train loss', train_loss)
        tf.scalar_summary('losses/test loss', test_loss)
        tf.scalar_summary('accuracy/train', slim.metrics.accuracy(preds_train, labels_train))
        tf.scalar_summary('accuracy/test', slim.metrics.accuracy(preds_test, labels_test))

        # Create training operation
        if FINE_TUNE:
            var2train = get_variables_to_train(trainable_scopes='fully_connected')
        else:
            var2train = get_variables_to_train()
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in var2train:
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        # Handle initialisation
        init_fn = None
        if FINE_TUNE:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fully_connected', ops.GraphKeys.GLOBAL_STEP])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training.
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=120, save_interval_secs=1200,
                            log_every_n_steps=100)
