import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver

from datasets import imagenet
from model_edge_2dis_128_1 import DCGAN
from preprocess import preprocess_image
from ops import *

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


def _get_variables_to_train(trainable_scopes=None):
    """Returns a list of variables to train.
    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


fine_tune = True
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'
BATCH_SIZE = 128
NUM_CLASSES = 1000
IM_SHAPE = [128, 128, 3]
MODEL_PATH = '/data/cvg/qhu/try_GAN/checkpoint_edge_twodis_128/050/DCGAN.model-80100'
LOG_DIR = '/data/cvg/simon/data/logs/ft1/'

# TODO: Indicate whether to use Keras or tensorflow model
tensorflow_model = True

sess = tf.Session()

if not tensorflow_model:
    from ToonNet import Classifier
    import keras.backend as K

    K.set_session(sess)
    myModel = Classifier((32, 32, 3), num_classes=NUM_CLASSES, num_layers=3, fine_tune=True)
    K.set_learning_phase(1)
    g = K.get_session().graph
else:
    def Classifier(inputs, fine_tune=False):
        # if fine_tune:
        model = DCGAN(sess, batch_size=BATCH_SIZE, is_train=not fine_tune, image_shape=IM_SHAPE)
        net = model.generator(inputs)
        # else:
        #     batch_norm_params = {
        #         'decay': 0.997,
        #         'epsilon': 1e-5,
        #         'scale': True,
        #         'updates_collections': tf.GraphKeys.UPDATE_OPS,
        #     }
        #     with slim.arg_scope([slim.conv2d],
        #                         kernel_size=[5, 5],
        #                         activation_fn=tf.nn.relu,
        #                         padding='SAME',
        #                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        #                         weights_regularizer=slim.l2_regularizer(0.0001),
        #                         normalizer_fn=slim.batch_norm,
        #                         normalizer_params=batch_norm_params):
        #         with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
        #             net = slim.conv2d(inputs, num_outputs=64, scope='conv_1', stride=2)
        #             net = slim.conv2d(net, num_outputs=128, scope='conv_2', stride=2)
        #             net = slim.conv2d(net, num_outputs=256, scope='conv_3', stride=2)
        #             net = slim.conv2d(net, num_outputs=512, scope='conv_4', stride=2)

        net = slim.flatten(net)
        net = slim.fully_connected(net, 2048, scope='fc1', activation_fn=tf.nn.relu)
        net = slim.dropout(net)
        net = slim.fully_connected(net, 2048, scope='fc2', activation_fn=tf.nn.relu)
        net = slim.dropout(net)
        return slim.fully_connected(net, NUM_CLASSES, scope='fc3', activation_fn=tf.nn.softmax), model


    g = tf.Graph()

with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

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
        image = preprocess_image(image, is_training=True, output_height=IM_SHAPE[0], output_width=IM_SHAPE[1])
        image = (tf.to_float(image) / 255. - 0.5) * 2.0

        images, labels = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=4,
            capacity=5 * BATCH_SIZE)

        labels = slim.one_hot_encoding(labels, NUM_CLASSES)

        # TODO: Create your model
        predictions, model = Classifier(images, fine_tune)

        # Define the loss
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
        tf.scalar_summary('losses/total loss', total_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

        # Create training operation
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=_get_variables_to_train())

        # See that moving average is also updated with g_optim.
        with tf.control_dependencies([train_op]):
            train_op = tf.group(tf.group(*batch_norm.assigners))

        init_fn = None
        if tensorflow_model and fine_tune:
            # TODO: Specify the layers of your model you want to exclude
            variables_to_restore = slim.get_variables_to_restore(exclude=['fc1', 'fc2', 'fc3', 'beta1_power', 'beta2_power'])
            init_fn = assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)

        # Start training.
        slim.learning.train(train_op, LOG_DIR, init_fn=init_fn, save_summaries_secs=300, save_interval_secs=600,
                            log_every_n_steps=10)
