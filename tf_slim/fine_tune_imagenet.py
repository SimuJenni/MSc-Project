from __future__ import print_function

from tensorflow.python.ops import control_flow_ops

from alexnet import alexnet
from datasets import imagenet
from preprocess import preprocess_image
from tf_slim.qhu_code.model_edge_advplus_128 import DCGAN
from tf_slim.qhu_code.ops import *
from utils import get_variables_to_train, assign_from_checkpoint_fn

slim = tf.contrib.slim

fine_tune = False
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 64
NUM_CLASSES = 1000
IM_SHAPE = [224, 224, 3]
PRE_TRAINED_SCOPE = 'generator'

MODEL_PATH = '/data/cvg/qhu/try_GAN/checkpoint_edge_advplus_128/010/DCGAN.model-148100'
LOG_DIR = '/data/cvg/simon/data/logs/alex_net_v2/'

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)


def Classifier(inputs, fine_tune=False):
    if fine_tune:
        # Specify model to fine-tune here
        model = DCGAN(sess, batch_size=BATCH_SIZE, is_train=True, image_shape=IM_SHAPE)
        with tf.variable_scope(PRE_TRAINED_SCOPE):
            net = model.discriminator(inputs)

        batch_norm_params = {'decay': 0.9997, 'epsilon': 0.001}
        with tf.variable_scope('fully_connected'):
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
                                           biases_initializer=tf.zeros_initializer, )
        return net
    else:
        net = alexnet(inputs, use_batch_norm=True)
        return net


g = tf.Graph()

with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        dataset = imagenet.get_split('train', DATA_DIR)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=8,
            common_queue_capacity=32 * BATCH_SIZE,
            common_queue_min=8 * BATCH_SIZE)

        [image, label] = provider.get(['image', 'label'])

        image = preprocess_image(image, is_training=True, output_height=IM_SHAPE[0], output_width=IM_SHAPE[1])
        if fine_tune:
            image = tf.cast(image, tf.float32) * (2. / 255) - 1

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=8,
            capacity=8 * BATCH_SIZE)

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # Create the model
        predictions = Classifier(images, fine_tune)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        total_loss = slim.losses.get_total_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates], total_loss)

        tf.scalar_summary('losses/total loss', total_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

        decay_steps = int(imagenet.SPLITS_TO_SIZES['train'] / BATCH_SIZE * 2.0)
        learning_rate = tf.train.exponential_decay(0.01,
                                                   global_step,
                                                   decay_steps,
                                                   0.94,
                                                   staircase=True,
                                                   name='exponential_decay_learning_rate')

        # Define optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, epsilon=1.0, momentum=0.9, decay=0.9)

        # Create training operation
        if fine_tune:
            var2train = get_variables_to_train(trainable_scopes='fully_connected')
        else:
            var2train = get_variables_to_train()

        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, summarize_gradients=True)

        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fc1', 'fc2', 'fc3'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training.
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
