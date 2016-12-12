import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib import slim as slim


def montage(imgs, num_h, num_w):
    """Makes a motage of imgs that can be used in image_summaries.

    Args:
        imgs: Tensor of images
        num_h: Number of images per column
        num_w: Number of images per row

    Returns:
        A montage of num_h*num_w images
    """
    imgs = tf.unpack(imgs)
    img_rows = [None] * num_h
    for r in range(num_h):
        img_rows[r] = tf.concat(1, imgs[r * num_w:(r + 1) * num_w])
    montage = tf.concat(0, img_rows)
    return tf.expand_dims(montage, 0)


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


def kl_divergence(mu1, lvar1, mu2, lvar2, scope=None):
    with ops.op_scope([mu1, lvar1, mu2, lvar2],
                      scope, "kl_divergence") as scope:
        pm = slim.flatten(mu1)
        qm = slim.flatten(mu2)
        pv = slim.flatten(lvar1)
        qv = slim.flatten(lvar2)
        # Determinants of diagonal covariances
        dpv = math_ops.reduce_sum(pv, reduction_indices=[1])
        dqv = math_ops.reduce_sum(qv, reduction_indices=[1])
        # Inverse of diagonal covariance
        iqv = 1. / qv
        # Difference between means pm, qm
        diff = qm - pm

        return math_ops.reduce_mean((0.5 *
                                     (dqv-dpv
                                      + math_ops.reduce_sum(math_ops.exp(pv-qv), reduction_indices=[1])
                                      + math_ops.reduce_sum(diff * math_ops.exp(-qv) * diff, reduction_indices=[1])
                                      - pm.get_shape().as_list()[1])))

        # return math_ops.reduce_mean((0.5 *
        #                              (math_ops.log((math_ops.div(dqv, dpv)))
        #                               + math_ops.reduce_sum(iqv * pv, reduction_indices=[1])
        #                               + math_ops.reduce_sum(diff * iqv * diff, reduction_indices=[1])
        #                               - pm.get_shape().as_list()[1])))


def kl_correct(mu, log_var, scope=None):
    with ops.op_scope([mu, log_var],
                      scope, "kl_divergence") as scope:
        return -0.5 * math_ops.reduce_sum(1.0 + log_var - math_ops.square(mu) - math_ops.exp(log_var))
