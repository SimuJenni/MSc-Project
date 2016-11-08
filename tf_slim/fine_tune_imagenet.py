import tensorflow as tf
from tensorflow.contrib.framework import assign_from_checkpoint_fn
from model_edge_2dis_128 import DCGAN

from datasets import imagenet
from constants import LOG_DIR

slim = tf.contrib.slim

DATA_DIR = 'data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 128


def Classifier(inputs):
    net = DCGAN()

    pass


g = tf.Graph()
with g.as_default():

    # Selects the 'validation' dataset.
    dataset = imagenet.get_split('train', DATA_DIR)

    # Creates a TF-Slim DataProvider which reads the dataset in the background
    # during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * BATCH_SIZE,
        common_queue_min=10 * BATCH_SIZE)

    [image, label] = provider.get(['image', 'label'])
    images, labels = tf.train.batch(
        [image, label],
        batch_size=BATCH_SIZE,
        num_threads=4,
        capacity=5 * BATCH_SIZE)

    labels = slim.one_hot_encoding(
        labels, dataset.num_classes)

    # TODO: Create the model
    predictions = myModel(images, is_training=True)

    # Define the loss
    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total loss', total_loss)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

    # Create training operation
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # TODO: Specify where the Model, trained on ImageNet, was saved.
    model_path = '/data/cvg/qhu/try_GAN/checkpoint_edge_twodis_128/028/DCGAN.model-100'

    # TODO: Specify where the new model will live:
    log_dir = LOG_DIR

    # TODO: Specify the layers of your model you want to exclude
    variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
    init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

    # Start training.
    slim.learning.train(train_op, log_dir, init_fn=init_fn)