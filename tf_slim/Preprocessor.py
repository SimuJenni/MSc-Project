from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, augment_color=False, aspect_ratio_range=(0.9, 1.1), area_range=(0.4, 1.0)):
        self.target_shape = target_shape
        self.augment_color = augment_color
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def process_train_toonnet(self, image, edge, cartoon):
        # Select random crops
        [image, edge, cartoon] = random_crop([image, edge, cartoon], self.target_shape[0], self.target_shape[1])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])
        edge.set_shape([self.target_shape[0], self.target_shape[1], 1])
        cartoon.set_shape([self.target_shape[0], self.target_shape[1], 3])

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.
        edge = tf.to_float(edge) / 255.
        cartoon = tf.to_float(cartoon) * (2. / 255.) - 1.

        # Flip left-right
        p = tf.random_uniform(shape=(), minval=0.0, maxval=1.0)
        image = flip_lr(image, p)
        edge = flip_lr(edge, p)
        cartoon = flip_lr(cartoon, p)

        return image, edge, cartoon

    def process_test_toonnet(self, image, edge, cartoon):
        # Select random crops
        [image, edge, cartoon] = central_crop([image, edge, cartoon], self.target_shape[0], self.target_shape[1])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])
        edge.set_shape([self.target_shape[0], self.target_shape[1], 1])
        cartoon.set_shape([self.target_shape[0], self.target_shape[1], 3])

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.
        edge = tf.to_float(edge) / 255.
        cartoon = tf.to_float(cartoon) * (2. / 255.) - 1.

        return image, edge, cartoon

    def process_transfer_train(self, image, thread_id=0):
        # Select random crops
        image = distort_image(image, self.target_shape[0], self.target_shape[1],
                              self.aspect_ratio_range, self.area_range)

        # Color and contrast augmentation
        image = tf.to_float(image) / 255.
        if self.augment_color:
            image = dist_color(image, thread_id)
            image = tf.clip_by_value(image, 0.0, 1.0)

        # Scale to [-1, 1]
        image = tf.to_float(image) * 2. - 1.

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image

    def process_transfer_test(self, image):
        # Crop the central region of the image with an area containing 85% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.85)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.target_shape[0], self.target_shape[1]], align_corners=False)
        image = tf.squeeze(image, [0])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.

        return image


class VOCPreprocessor(Preprocessor):
    def __init__(self, target_shape, augment_color=True, aspect_ratio_range=(0.9, 1.1), area_range=(0.1, 1.0)):
        Preprocessor.__init__(self, target_shape, augment_color, aspect_ratio_range, area_range)

    def process_transfer_test(self, image):
        # Select random crops
        image = distort_image(image, self.target_shape[0], self.target_shape[1],
                              self.aspect_ratio_range, self.area_range)

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image


class ImageNetPreprocessor(Preprocessor):
    def __init__(self, target_shape, augment_color=False):
        Preprocessor.__init__(self, target_shape, augment_color)

    def process_transfer_test(self, image):
        image.set_shape([256, 256, 3])
        image = tf.image.resize_image_with_crop_or_pad(image, self.target_shape[0], self.target_shape[1])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.

        return image

    def process_transfer_train(self, image, thread_id=0):
        # Select random crops
        image = tf.random_crop(image, size=(self.target_shape[0], self.target_shape[1], 3))

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image


def distort_image(image, height, width, aspect_ratio_range=(0.9, 1.1), area_range=(0.1, 1.0)):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        [[[0, 0, 1, 1]]],
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.expand_dims(distorted_image, 0)
    resized_image = tf.image.resize_bilinear(distorted_image, [height, width], align_corners=False)
    distorted_image = tf.squeeze(resized_image)
    distorted_image.set_shape([height, width, 3])
    return distorted_image


def dist_color(image, thread_id):
    color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=16. / 255.)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.1)

    return image


def flip_lr(image, p):
    return tf.cond(p > 0.5, fn1=lambda: image, fn2=lambda: tf.image.flip_left_right(image))


def crop(image, offset_height, offset_width, crop_height, crop_width):
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

    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies([rank_assertion],
                                                       tf.pack([crop_height, crop_width, original_shape[2]]))
    size_assertion = tf.Assert(tf.logical_and(tf.greater_equal(original_shape[0], crop_height),
                                              tf.greater_equal(original_shape[1], crop_width)),
                               ['Crop size greater than the image size.'])
    offsets = tf.to_int32(tf.pack([offset_height, offset_width, 0]))
    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to define the crop size.
    image = control_flow_ops.with_dependencies([size_assertion], tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.
    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:
      image, depths, normals = _random_crop([image, depths, normals], 120, 150)
    Args:
      image_list: a list of image tensors of the same dimension but possibly varying channel.
      crop_height: the new height.
      crop_width: the new width.
    Returns:
      the image_list with cropped images.
    Raises:
      ValueError: if there are multiple image inputs provided with different size or the images are smaller than the
       crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                ['Wrong rank for tensor  %s [expected] [actual]',
                                 image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies([rank_assertions[0]], tf.shape(image_list[0]))
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(tf.logical_and(tf.greater_equal(image_height, crop_height),
                                                tf.greater_equal(image_width, crop_width)),
                                 ['Crop size greater than the image size.'])
    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies([rank_assertions[i]], tf.shape(image))
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(tf.equal(height, image_height), ['Wrong height for tensor %s [expected][actual]',
                                                                   image.name, height, image_height])
        width_assert = tf.Assert(tf.equal(width, image_width), ['Wrong width for tensor %s [expected][actual]',
                                                                image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [crop(image, offset_height, offset_width,
                 crop_height, crop_width) for image in image_list]


def central_crop(image_list, crop_height, crop_width):
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

        outputs.append(crop(image, offset_height, offset_width,
                            crop_height, crop_width))
    return outputs