import tensorflow as tf

from tf_slim.datasets import imagenet
from model_edge_2dis_128 import DCGAN


slim = tf.contrib.slim

DATA_DIR = 'data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 256
NUM_CLASSES = 1000

# TODO: Set path for storing the model
log_dir = '/Users/simujenni/MSc-Project/data/logs/'

# TODO: Indicate whether to use Keras or tensorflow model
tensorflow_model = False

sess = tf.Session()

if not tensorflow_model:
    from ToonNet import Classifier
    import keras.backend as K
    K.set_session(sess)
    myModel = Classifier((32, 32, 3), num_classes=NUM_CLASSES, num_layers=3, fine_tune=True)
    K.set_learning_phase(1)
    g = K.get_session().graph
else:
    model_path = '/data/cvg/qhu/try_GAN/checkpoint_edge_twodis_128/028/DCGAN.model-100'
    def Classifier(inputs):
        model = DCGAN(sess, batch_size=BATCH_SIZE, is_train=False)
        net = model.generator(inputs)
        net = slim.fully_connected(net, 2048, scope='fc1', activation_fn=tf.nn.relu)
        net = slim.dropout(net)
        net = slim.fully_connected(net, 2048, scope='fc2', activation_fn=tf.nn.relu)
        net = slim.dropout(net)
        return slim.fully_connected(net, NUM_CLASSES, scope='fc3', activation_fn=tf.nn.softmax)
    g = tf.Graph()

with sess.as_default():
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

        # TODO: Adjust preprocessing of images
        image = (tf.to_float(image) / 255. - 0.5) * 2.0

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=4,
            capacity=5 * BATCH_SIZE)

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # TODO: Create your model
        predictions = myModel(images)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

        # Create training operation
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        init_fn = None
        if tensorflow_model:
            from tensorflow.contrib.framework import assign_from_checkpoint_fn
            # TODO: Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(exclude=['fc1', 'fc2', 'fc3'])
            init_fn = assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Start training.
        slim.learning.train(train_op, log_dir, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=600)
