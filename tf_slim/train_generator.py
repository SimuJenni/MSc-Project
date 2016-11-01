import tensorflow as tf
from ToonNet import ToonGenerator
import imagenet_toon
from constants import DATA_DIR, LOG_DIR

slim = tf.contrib.slim

BATCH_SIZE = 128

train_log_dir = LOG_DIR
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

g = tf.Graph()
with g.as_default():
    # Get the dataset
    dataset = imagenet_toon.get_split('train', DATA_DIR)

    # Creates a TF-Slim DataProvider which reads the dataset in the background
    # during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * BATCH_SIZE,
        common_queue_min=10 * BATCH_SIZE)

    # Create mini-batch
    [X, Y] = provider.get(['image_cartoon', 'image_original'])
    Xs, Ys = tf.train.batch(
        [X, Y],
        batch_size=BATCH_SIZE,
        num_threads=4,
        capacity=5 * BATCH_SIZE)

    # Create model
    Ys_pred = ToonGenerator(Xs)

    # Define the loss
    slim.losses.softmax_cross_entropy(Ys_pred, Ys)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total loss', total_loss)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

    # Create training operation
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Run training
    slim.learning.train(train_op, train_log_dir)
