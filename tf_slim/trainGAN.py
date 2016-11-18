from __future__ import print_function

from tensorflow.python.ops import control_flow_ops

from datasets import imagenet
from ops import *
from preprocess import preprocess_image
from tf_slim.utils import assign_from_checkpoint_fn, get_variables_to_train

slim = tf.contrib.slim

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
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=8,
            common_queue_capacity=32 * BATCH_SIZE,
            common_queue_min=8 * BATCH_SIZE)

        [image, label] = provider.get(['image', 'label'])

        image = preprocess_image(image, is_training=True, output_height=IM_SHAPE[0], output_width=IM_SHAPE[1])

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=8,
            capacity=8 * BATCH_SIZE)

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # Create the model
        predictions = Classifier(images, fine_tune, use_batch_norm=use_bn)

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

        decay_steps = int(imagenet._SPLITS_TO_SIZES['train'] / BATCH_SIZE * 2.0)
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
                exclude=['fc1', 'fc2', 'fc3', 'beta1_power', 'beta2_power'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training.
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
