from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from ToonNet import ToonGenAE
from datasets import cifar10
from preprocess import preprocess_image
from tf_slim.ToonNet import classifier
from utils import get_variables_to_train, assign_from_checkpoint_fn

slim = tf.contrib.slim


def model(inputs):
    return ToonGenAE(inputs, num_layers=4)


fine_tune = False
data = cifar10
BATCH_SIZE = 64
NUM_CLASSES = 1000
NUM_EPOCHS = 30
IM_SHAPE = [224, 224, 3]

MODEL_PATH = '/data/cvg/simon/data/logs/cifar10_gan/model.ckpt-9750'
LOG_DIR = '/data/cvg/simon/data/logs/cifar10_finetune/'

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # Get the dataset
        dataset = data.get_split('train')
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=8,
            common_queue_capacity=32 * BATCH_SIZE,
            common_queue_min=8 * BATCH_SIZE)
        [image, label] = provider.get(['image', 'label'])

        # Pre-process images
        image = preprocess_image(image, is_training=True,
                                 output_height=IM_SHAPE[0], output_width=IM_SHAPE[1],
                                 resize_side_min=IM_SHAPE[0],
                                 resize_side_max=int(IM_SHAPE[0] * 1.5))
        image = tf.cast(image, tf.float32) * (2. / 255) - 1

        # Make batches
        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=8,
            capacity=8 * BATCH_SIZE)
        labels = slim.one_hot_encoding(labels, data.NUM_CLASSES)

        # Create the model
        predictions = classifier(images, model=model, num_classes=data.NUM_CLASSES)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates], total_loss)

        # Gather all summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))
        tf.scalar_summary('losses/total loss', total_loss)
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

        # Define learning rate
        decay_steps = int(data.SPLITS_TO_SIZES['train'] / BATCH_SIZE * 2.0)
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

        # Handle initialisation
        init_fn = None
        if fine_tune:
            # Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fc1', 'fc2', 'fc3'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training
        num_train_steps = data.SPLITS_TO_SIZES['train'] / BATCH_SIZE * NUM_EPOCHS
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
