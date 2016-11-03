import tensorflow as tf
from ToonNet import Classifier
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from tf_slim.datasets import cifar10
from constants import LOG_DIR

slim = tf.contrib.slim

DATA_DIR = '/data/cvg/simon/data/cifar-10-TFRecords/'
BATCH_SIZE = 256
NUM_CLASSES = 10

# TODO: Specify where the new model will live:
log_dir = LOG_DIR
load_weights_tf = False

with sess.as_default():

    g = tf.Graph()
    with g.as_default():

        # Selects the 'validation' dataset.
        dataset = cifar10.get_split('train', DATA_DIR)

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

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # TODO: Create the model
        # predictions = myModel(images, is_training=True)
        model = Classifier((32, 32, 3), num_classes=NUM_CLASSES, num_layers=3, fine_tune=True)
        predictions = model(images)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        tf.summary.scalar('losses/total loss', total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

        # Create training operation
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        if load_weights_tf:
            # TODO: Specify where the Model, trained on ImageNet, was saved.
            model_path = '/path/to/pre_trained_model.checkpoint'

            # TODO: Specify the layers of your model you want to exclude
            from tensorflow.contrib.framework import assign_from_checkpoint_fn
            variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
            init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

            # Start training.
            slim.learning.train(train_op, log_dir, init_fn=init_fn)

        else:
            slim.learning.train(train_op, log_dir)
