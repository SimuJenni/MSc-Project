import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
import numpy as np


def per_image_standardization(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  `stddev` is the standard deviation of all values in `image`. It is capped
  away from zero to protect against division by 0 when handling uniform images.

  Args:
    image: 3-D tensor of shape `[height, width, channels]`.

  Returns:
    The standardized image with same shape as `image`.

  Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
  """
  image = ops.convert_to_tensor(image, name='image')
  num_pixels = math_ops.reduce_prod(array_ops.shape(image))

  image = math_ops.cast(image, dtype=dtypes.float32)
  image_mean = math_ops.reduce_mean(image)

  variance = (math_ops.reduce_mean(math_ops.square(image)) -
              math_ops.square(image_mean))
  variance = gen_nn_ops.relu(variance)
  stddev = math_ops.sqrt(variance)

  # Apply a minimum normalization that protects us against uniform images.
  min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
  pixel_value_scale = math_ops.maximum(stddev, min_stddev)
  pixel_value_offset = image_mean

  image = math_ops.sub(image, pixel_value_offset)
  image = math_ops.div(image, pixel_value_scale)
  return image


def weights_montage(weights, grid_Y, grid_X, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
        weights: tensor of shape [Y, X, NumChannels, NumKernels]
        (grid_Y, grid_X): shape of the grid. Require: NumKernels == grid_Y * grid_X
        pad: number of black pixels around each filter (between them)

    Return:
        Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    """

    # x_min = tf.reduce_min(weights)
    # x_max = tf.reduce_max(weights)
    #
    # weights1 = (weights - x_min) / (x_max - x_min)
    weights1 = tf.map_fn(lambda img: tf.image.per_image_standardization(img), weights)

    # pad X and Y
    x1 = tf.pad(weights1, tf.constant([[pad, pad], [pad, pad], [1, 1], [1, 1]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = weights1.get_shape()[0] + 2 * pad
    X = weights1.get_shape()[1] + 2 * pad

    channels = weights1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))  # 3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))  # 3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def montage_tf(imgs, num_h, num_w):
    """Makes a montage of imgs that can be used in image_summaries.

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


def montage(images, gray=False):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Args:
        images (np.ndarray): Images batch x height x width x channels
        saveto (str): Location to save the resulting montage image.

    Returns:
        m (np.ndarray): Montage image.
    """
    from PIL import Image

    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    if gray:
        m = Image.fromarray((m * 255).astype(dtype=np.uint8))
    return m
