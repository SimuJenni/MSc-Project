from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops

import numpy as np

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _flip_lr(image, p):
    return tf.cond(p > 0.5, fn1=lambda: image, fn2=lambda: tf.image.flip_left_right(image))


def rotate_image_tensor(image, angle, mode='black'):
    """
    Rotates a 3D tensor (HWD), which represents an image by given radian angle.

    New image has the same size as the input image.

    mode controls what happens to border pixels.
    mode = 'black' results in black bars (value 0 in unknown areas)
    mode = 'white' results in value 255 in unknown areas
    mode = 'ones' results in value 1 in unknown areas
    mode = 'repeat' keeps repeating the closest pixel known
    """
    s = image.get_shape().as_list()
    assert len(s) == 3, "Input needs to be 3D."
    assert (mode == 'repeat') or (mode == 'black') or (mode == 'white') or (mode == 'ones'), "Unknown boundary mode."
    image_center = [np.floor(x / 2) for x in s]

    # Coordinates of new image
    coord1 = tf.range(s[0])
    coord2 = tf.range(s[1])

    # Create vectors of those coordinates in order to vectorize the image
    coord1_vec = tf.tile(coord1, [s[1]])

    coord2_vec_unordered = tf.tile(coord2, [s[0]])
    coord2_vec_unordered = tf.reshape(coord2_vec_unordered, [s[0], s[1]])
    coord2_vec = tf.reshape(tf.transpose(coord2_vec_unordered, [1, 0]), [-1])

    # center coordinates since rotation center is supposed to be in the image center
    coord1_vec_centered = coord1_vec - image_center[0]
    coord2_vec_centered = coord2_vec - image_center[1]

    coord_new_centered = tf.cast(tf.pack([coord1_vec_centered, coord2_vec_centered]), tf.float32)

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    rot_mat_inv = tf.cast(rot_mat_inv, tf.float32)
    coord_old_centered = tf.matmul(rot_mat_inv, coord_new_centered)

    # Find nearest neighbor in old image
    coord1_old_nn = tf.cast(tf.round(coord_old_centered[0, :] + image_center[0]), tf.int32)
    coord2_old_nn = tf.cast(tf.round(coord_old_centered[1, :] + image_center[1]), tf.int32)

    # Clip values to stay inside image coordinates
    if mode == 'repeat':
        coord_old1_clipped = tf.minimum(tf.maximum(coord1_old_nn, 0), s[0] - 1)
        coord_old2_clipped = tf.minimum(tf.maximum(coord2_old_nn, 0), s[1] - 1)
    else:
        outside_ind1 = tf.logical_or(tf.greater(coord1_old_nn, s[0] - 1), tf.less(coord1_old_nn, 0))
        outside_ind2 = tf.logical_or(tf.greater(coord2_old_nn, s[1] - 1), tf.less(coord2_old_nn, 0))
        outside_ind = tf.logical_or(outside_ind1, outside_ind2)

        coord_old1_clipped = tf.boolean_mask(coord1_old_nn, tf.logical_not(outside_ind))
        coord_old2_clipped = tf.boolean_mask(coord2_old_nn, tf.logical_not(outside_ind))

        coord1_vec = tf.boolean_mask(coord1_vec, tf.logical_not(outside_ind))
        coord2_vec = tf.boolean_mask(coord2_vec, tf.logical_not(outside_ind))

    coord_old_clipped = tf.cast(tf.transpose(tf.pack([coord_old1_clipped, coord_old2_clipped]), [1, 0]), tf.int32)

    # Coordinates of the new image
    coord_new = tf.transpose(tf.cast(tf.pack([coord1_vec, coord2_vec]), tf.int32), [1, 0])

    image_channel_list = tf.split(2, s[2], image)

    image_rotated_channel_list = list()
    for image_channel in image_channel_list:
        image_chan_new_values = tf.gather_nd(tf.squeeze(image_channel), coord_old_clipped)

        if (mode == 'black') or (mode == 'repeat'):
            background_color = 0
        elif mode == 'ones':
            background_color = 1
        elif mode == 'white':
            background_color = 255

        image_rotated_channel_list.append(tf.sparse_to_dense(coord_new, [s[0], s[1]], image_chan_new_values,
                                                             background_color, validate_indices=False))

    image_rotated = tf.transpose(tf.pack(image_rotated_channel_list), [1, 2, 0])

    return image_rotated


def adjust_gamma(image, gamma=1, gain=1):
    """Performs Gamma Correction on the input image.
      Also known as Power Law Transform. This function transforms the
      input image pixelwise according to the equation Out = In**gamma
      after scaling each pixel to the range 0 to 1.

    Args:
      image : A Tensor.
      gamma : A scalar. Non negative real number.
      gain  : A scalar. The constant multiplier.

    Returns:
      A Tensor. Gamma corrected output image.

    Notes:
      For gamma greater than 1, the histogram will shift towards left and
      the output image will be darker than the input image.
      For gamma less than 1, the histogram will shift towards right and
      the output image will be brighter than the input image.

    References:
      [1] http://en.wikipedia.org/wiki/Gamma_correction
    """

    with ops.op_scope([image, gamma, gain], None, 'adjust_gamma') as name:
        # Convert pixel value to DT_FLOAT for computing adjusted image
        img = ops.convert_to_tensor(image, name='img', dtype=dtypes.float32)
        # Keep image dtype for computing the scale of corresponding dtype
        image = ops.convert_to_tensor(image, name='image')

        # scale = max(dtype) - min(dtype)
        scale = constant_op.constant(2, dtype=dtypes.float32)
        # According to the definition of gamma correction
        adjusted_img = (img / scale) ** gamma * scale * gain

        return adjusted_img


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.pack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.pack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.
    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:
      image, depths, normals = _random_crop([image, depths, normals], 120, 150)
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.
    Returns:
      the image_list with cropped images.
    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0]))
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                                   tf.shape(image))
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
    Returns:
      the centered image.
    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(2, num_channels, image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(2, channels)


def _smallest_size_at_least(height, width, smallest_side):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side, num_channels=3):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image = tf.cond(tf.rank(resized_image) < 3,
                            fn1=lambda: tf.expand_dims(resized_image, 2),
                            fn2=lambda: resized_image)
    resized_image.set_shape([None, None, num_channels])
    return resized_image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         resize_side_min=_RESIZE_SIDE_MIN,
                         resize_side_max=_RESIZE_SIDE_MAX):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
      [`resize_size_min`, `resize_size_max`].
    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing.
    Returns:
      A preprocessed image.
    """
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_for_eval(image, output_height, output_width, resize_side):
    """Preprocesses the given image for evaluation.
    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.        num_train_steps = (imagenet.SPLITS_TO_SIZES['train'] / BATCH_SIZE) * NUM_EP

      resize_side: The smallest side of the image for aspect-preserving resizing.
    Returns:
      A preprocessed image.
    """
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocess_toon_train(image, edge, cartoon, output_height, output_width, resize_side_min=_RESIZE_SIDE_MIN,
                          resize_side_max=_RESIZE_SIDE_MAX):
    # Compute zoom side-size
    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    # Resize/zoom
    image = _aspect_preserving_resize(image, resize_side)
    edge = _aspect_preserving_resize(edge, resize_side, num_channels=1)
    cartoon = _aspect_preserving_resize(cartoon, resize_side)

    # Select random crops
    [image, edge, cartoon] = _random_crop([image, edge, cartoon], output_height, output_width)

    # Resize to output size
    image.set_shape([output_height, output_width, 3])
    edge.set_shape([output_height, output_width, 1])
    cartoon.set_shape([output_height, output_width, 3])

    # Scale to [-1, 1]
    image = tf.to_float(image) * (2. / 255.) - 1.
    edge = tf.to_float(edge) / 255.
    cartoon = tf.to_float(cartoon) * (2. / 255.) - 1.

    # Flip left-right
    p = tf.random_uniform(shape=(), minval=0.0, maxval=1.0)
    image = _flip_lr(image, p)
    edge = _flip_lr(edge, p)
    cartoon = _flip_lr(cartoon, p)

    return image, edge, cartoon


def preprocess_toon_test(image, edge, cartoon, output_height, output_width, resize_side=_RESIZE_SIDE_MIN):
    # Resize/zoom
    image = _aspect_preserving_resize(image, resize_side)
    edge = _aspect_preserving_resize(edge, resize_side, num_channels=1)
    cartoon = _aspect_preserving_resize(cartoon, resize_side)

    # Select random crops
    [image, edge, cartoon] = _central_crop([image, edge, cartoon], output_height, output_width)

    # Resize to output size
    image.set_shape([output_height, output_width, 3])
    edge.set_shape([output_height, output_width, 1])
    cartoon.set_shape([output_height, output_width, 3])

    # Scale to [-1, 1]
    image = tf.to_float(image) * (2. / 255.) - 1.
    edge = tf.to_float(edge) / 255.
    cartoon = tf.to_float(cartoon) * (2. / 255.) - 1.

    return image, edge, cartoon


def preprocess_finetune_train(image, edge, output_height, output_width, resize_side_min=_RESIZE_SIDE_MIN,
                              resize_side_max=_RESIZE_SIDE_MAX):
    # Randomly rotate
    angle = np.random.rand(1) * 0.3 - 0.15
    image = rotate_image_tensor(image, angle, mode='black')
    edge = rotate_image_tensor(edge, angle, mode='black')

    # Compute zoom side-size
    resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

    # Resize/zoom
    image = _aspect_preserving_resize(image, resize_side)
    edge = _aspect_preserving_resize(edge, resize_side, num_channels=1)

    # Select random crops
    [image, edge] = _random_crop([image, edge], output_height, output_width)

    # Resize to output size
    image.set_shape([output_height, output_width, 3])
    edge.set_shape([output_height, output_width, 1])

    image = tf.to_float(image) / 255.
    image = tf.image.random_hue(image, 0.05, seed=None)
    image = tf.image.random_saturation(image, 0.7, 1.4, seed=None)
    image = tf.image.random_contrast(image, 0.7, 1.4, seed=None)
    image = tf.image.random_brightness(image, 0.2, seed=None)

    # Scale to [-1, 1]
    image = tf.to_float(image) * 2. - 1.
    edge = tf.to_float(edge) / 255.

    # Flip left-right
    p = tf.random_uniform(shape=(), minval=0.0, maxval=1.0)
    image = _flip_lr(image, p)
    edge = _flip_lr(edge, p)

    return image, edge


def preprocess_finetune_test(image, edge, output_height, output_width, resize_side=_RESIZE_SIDE_MIN):
    # Resize/zoom
    image = _aspect_preserving_resize(image, resize_side)
    edge = _aspect_preserving_resize(edge, resize_side, num_channels=1)

    # Select random crops
    [image, edge] = _central_crop([image, edge], output_height, output_width)

    # Resize to output size
    image.set_shape([output_height, output_width, 3])
    edge.set_shape([output_height, output_width, 1])

    # Scale to [-1, 1]
    image = tf.to_float(image) * (2. / 255.) - 1.
    edge = tf.to_float(edge) / 255.

    return image, edge


def preprocess_image(image, output_height, output_width, is_training=False,
                     resize_side_min=_RESIZE_SIDE_MIN,
                     resize_side_max=_RESIZE_SIDE_MAX):
    """Preprocesses the given image.
    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
        ignored. Otherwise, the resize side is sampled from
          [resize_size_min, resize_size_max].
    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, output_height, output_width,
                                    resize_side_min, resize_side_max)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   resize_side_min)
