from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver

from alexnet import alexnet_v2
from datasets import imagenet
from model_edge_2dis_128_1 import DCGAN
from ops import *
from preprocess import preprocess_image

slim = tf.contrib.slim


def assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=False,
                              reshape_variables=False):
    """Returns a function that assigns specific variables from a checkpoint.
    Args:
      model_path: The full path to the model checkpoint. To get latest checkpoint
          use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
      var_list: A list of `Variable` objects or a dictionary mapping names in the
          checkpoint to the correspoing variables to initialize. If empty or None,
          it would return  no_op(), None.
      ignore_missing_vars: Boolean, if True it would ignore variables missing in
          the checkpoint with a warning instead of failing.
      reshape_variables: Boolean, if True it would automatically reshape variables
          which are of different shape then the ones stored in the checkpoint but
          which have the same number of elements.
    Returns:
      A function that takes a single argument, a `tf.Session`, that applies the
      assignment operation.
    Raises:
      ValueError: If the checkpoint specified at `model_path` is missing one of
        the variables in `var_list`.
    """
    if ignore_missing_vars:
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
        if isinstance(var_list, dict):
            var_dict = var_list
        else:
            var_dict = {var.op.name: var for var in var_list}
        available_vars = {}
        for var in var_dict:

            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                logging.warning(
                    'Variable %s missing in checkpoint %s', var, model_path)
        var_list = available_vars
    saver = tf_saver.Saver(var_list, reshape=reshape_variables)

    def callback(session):
        saver.restore(session, model_path)

    return callback


def get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        variables_to_train = tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

    print('Variables to train: {}'.format([v.op.name for v in variables_to_train]))

    return variables_to_train


fine_tune = True
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 128
NUM_CLASSES = 1000
IM_SHAPE = [224, 224, 3]
IM_SHAPE = [128, 128, 3]

MODEL_PATH = '/data/cvg/qhu/try_GAN/checkpoint_edge_twodis_128/050/DCGAN.model-80100'
LOG_DIR = '/data/cvg/simon/data/logs/alex_net/'
LOG_DIR = '/data/cvg/simon/data/logs/fine_tune/'

# TODO: Indicate whether to use Keras or tensorflow model
tensorflow_model = True

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.INFO)

if not tensorflow_model:
    from ToonNet import Classifier
    import keras.backend as K

    K.set_session(sess)
    myModel = Classifier((32, 32, 3), num_classes=NUM_CLASSES, num_layers=3, fine_tune=True)
    K.set_learning_phase(1)
    g = K.get_session().graph
else:

    def Classifier(inputs, fine_tune=False):
        if fine_tune:
            model = DCGAN(sess, batch_size=BATCH_SIZE, is_train=not fine_tune, image_shape=IM_SHAPE)
            with tf.variable_scope('generator') as scope:
                net = model.generator(inputs)
            with tf.variable_scope('fully_connected') as scope:
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, 4096, scope='fc1')
                    net = slim.dropout(net)
                    net = slim.fully_connected(net, 4096, scope='fc2')
                    net = slim.dropout(net)
                    net = slim.fully_connected(net, NUM_CLASSES, scope='fc3', activation_fn=None)
            return net
        else:
            net = alexnet_v2(inputs)
            return net


    g = tf.Graph()

with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        # # Selects the 'validation' dataset.
        # data_files = imagenet.get_datafiles('train', DATA_DIR)
        # images, labels = batch_inputs(data_files=data_files, batch_size=BATCH_SIZE, train=True, num_preprocess_threads=4)

        dataset = imagenet.get_split('train', DATA_DIR)
        # Creates a TF-Slim DataProvider which reads the dataset in the background
        # during both training and testing.
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=16,
            common_queue_capacity=16 * BATCH_SIZE,
            common_queue_min=8 * BATCH_SIZE)

        [image, label] = provider.get(['image', 'label'])

        # TODO: Adjust preprocessing of images
        image = preprocess_image(image, is_training=True, output_height=IM_SHAPE[0], output_width=IM_SHAPE[1])

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=8,
            capacity=8 * BATCH_SIZE)

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # TODO: Create your model
        predictions = Classifier(images, fine_tune)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.histogram_summary(variable.op.name, variable))

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', accuracy)

        with tf.name_scope('label'):
            tf.histogram_summary('label', tf.argmax(labels, 1))
            tf.histogram_summary('prediction', tf.argmax(predictions, 1))
            tf.histogram_summary('logits', predictions)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer()

        # Create training operation
        if fine_tune:
            var2train = get_variables_to_train(trainable_scopes='fully_connected')
        else:
            var2train = get_variables_to_train()

        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=var2train,
                                                 global_step=global_step, summarize_gradients=True)

        init_fn = None
        if tensorflow_model and fine_tune:
            # TODO: Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(
                exclude=['fc1', 'fc2', 'fc3', 'beta1_power', 'beta2_power'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training.
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=3000,
                            log_every_n_steps=100)
