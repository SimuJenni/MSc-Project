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


def kl_divergence(mu1, sigma1, mu2, sigma2, epsilon=1e-8, scope=None):
    with ops.op_scope([mu1, sigma1, mu2, sigma2],
                      scope, "kl_divergence") as scope:
        mu1 = slim.flatten(mu1)
        mu2 = slim.flatten(mu2)
        sigma1 = slim.flatten(sigma1)
        sigma2 = slim.flatten(sigma2)
        k = mu1.get_shape().as_list()[1]
        t1 = math_ops.reduce_sum(math_ops.div(sigma1+math_ops.square(mu2-mu1), sigma2+epsilon), reduction_indices=[1])
        t2 = math_ops.log(math_ops.reduce_prod(math_ops.div(sigma2, sigma1+epsilon), reduction_indices=[1]))
        return math_ops.reduce_mean(0.5*(t1+t2-k))


def kl_correct(mu, log_var, scope=None):
    with ops.op_scope([mu, log_var],
                      scope, "kl_divergence") as scope:
        return -0.5 * math_ops.reduce_sum(1.0 + log_var - math_ops.square(mu) - math_ops.exp(log_var))

