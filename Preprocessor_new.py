from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, augment_color=False, aspect_ratio_range=(0.8, 1.2), area_range=(0.2, 1.0)):
        self.target_shape = target_shape
        self.augment_color = augment_color
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def central_crop(self, image):
        # Crop the central region of the image with an area containing 85% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.85)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.target_shape[0], self.target_shape[1]], align_corners=False)
        image = tf.squeeze(image, [0])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])
        return image

    def extract_and_resize_bbox(self, img, bbox_begin, bbox_size):
        # Crop the image to the specified bounding box.
        image = tf.slice(img, bbox_begin, bbox_size)
        image = tf.expand_dims(image, 0)
        resized_image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            fn1=lambda: tf.image.resize_bilinear(image, self.target_shape[:2], align_corners=False),
            fn2=lambda: tf.image.resize_bicubic(image, self.target_shape[:2], align_corners=False))
        image = tf.squeeze(resized_image)
        image.set_shape(self.target_shape)
        return image

    def process_train_toonnet(self, image, cartoon):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            [144, 144, 3],
            [[[0, 0, 1, 1]]],
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.area_range,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        image = self.extract_and_resize_bbox(image, bbox_begin, bbox_size)
        cartoon = self.extract_and_resize_bbox(cartoon, bbox_begin, bbox_size)

        image = tf.to_float(image) / 255.
        cartoon = tf.to_float(cartoon) / 255.

        if self.augment_color:
            bright_delta, sat, hue_delta, cont = sample_color_params()
            image = dist_color(image, bright_delta, sat, hue_delta, cont)
            image = tf.clip_by_value(image, 0.0, 1.0)
            cartoon = dist_color(cartoon, bright_delta, sat, hue_delta, cont)
            cartoon = tf.clip_by_value(cartoon, 0.0, 1.0)

        # Scale to [-1, 1]
        image = tf.to_float(image) * 2. - 1.
        cartoon = tf.to_float(cartoon) * 2. - 1.

        # Flip left-right
        p = tf.random_uniform(shape=(), minval=0.0, maxval=1.0)
        image = flip_lr(image, p)
        cartoon = flip_lr(cartoon, p)

        return image, cartoon

    def process_test_toonnet(self, image, cartoon):
        image = self.central_crop(image)
        cartoon = self.central_crop(cartoon)

        # Scale to [-1, 1]
        image = tf.to_float(image) * (2. / 255.) - 1.
        cartoon = tf.to_float(cartoon) * (2. / 255.) - 1.

        return image, cartoon

    def process_transfer_train(self, image, thread_id=0):
        # Select random crops
        image = distort_image(image, self.target_shape[0], self.target_shape[1],
                              self.aspect_ratio_range, self.area_range)

        # Color and contrast augmentation
        image = tf.to_float(image) / 255.
        if self.augment_color:
            image = dist_color_random(image, thread_id)
            image = tf.clip_by_value(image, 0.0, 1.0)

        # Scale to [-1, 1]
        image = tf.to_float(image) * 2. - 1.

        # Flip left-right
        image = tf.image.random_flip_left_right(image)

        return image

    def process_transfer_test(self, image):
        image = self.central_crop(image)

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


def sample_color_params(bright_max_delta=16./255., lower_sat=0.8, upper_sat=1.2, hue_max_delta=0.05, lower_cont=0.8,
                        upper_cont=1.2):
    bright_delta = tf.random_uniform([], -bright_max_delta, bright_max_delta)
    sat = tf.random_uniform([], lower_sat, upper_sat)
    hue_delta = tf.random_uniform([], -hue_max_delta, hue_max_delta)
    cont = tf.random_uniform([], lower_cont, upper_cont)
    return bright_delta, sat, hue_delta, cont


def dist_color_random(image, thread_id):
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


def dist_color(image, bright_delta, sat, hue_delta, cont, thread_id=0):
    color_ordering = thread_id % 2
    if color_ordering == 0:
        image = tf.image.adjust_brightness(image, delta=bright_delta)
        image = tf.image.adjust_saturation(image, saturation_factor=sat)
        image = tf.image.adjust_hue(image, delta=hue_delta)
        image = tf.image.adjust_contrast(image, contrast_factor=cont)
    elif color_ordering == 1:
        image = tf.image.adjust_brightness(image, delta=bright_delta)
        image = tf.image.adjust_contrast(image, contrast_factor=cont)
        image = tf.image.adjust_saturation(image, saturation_factor=sat)
        image = tf.image.adjust_hue(image, delta=hue_delta)

    return image


def flip_lr(image, p):
    return tf.cond(p > 0.5, fn1=lambda: image, fn2=lambda: tf.image.flip_left_right(image))
