import re, os
import tensorflow as tf
import numpy as np
import glob

def get_files(folder, file_regexp):
  files = []
  for root, dirnames, filenames in os.walk(folder):
      for filename in filenames:
        if re.match(file_regexp, filename):
          files.append(os.path.join(root, filename))
  return sorted(files)

def get_files_cached(folder, type, regexp, reload=False):
  f = os.path.join(folder, '{}_files.txt'.format(type))
  if os.path.isfile(f) and not reload:
    return [filename.strip() for filename in open(f).readlines()]
  else:
    filelist = get_files(folder, regexp)
    with open(f, 'w') as file_write:
      file_write.write('\n'.join(filelist))
    return filelist


def get_rendered_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'rendered', '.*r_\d{3}_' + size_suffix + '\.png$')


def get_albedo_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'albedo', '.*r_\d{3}_albedo\.png0001_' + size_suffix + '\.png$')


def get_normal_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'normal', '.*r_\d{3}_normal\.png0001_' + size_suffix + '\.png$')


def get_depth_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'depth', '.*r_\d{3}_depth\.png0001_' + size_suffix + '\.png$')


def get_sketch_files(folder, size_suffix='64x64'):
  return get_files_cached(folder, 'sketch', '.*r_\d{3}_sketch_' + size_suffix + '.png$')


def preprocess(image_tensor, img_size,resize_size, whiten='default', color=False,
               augment=True, augment_color=False, augment_translation=False):
  # Use same seed for flipping for every tensor, so they'll be flipped the same.
  seed = 42
  # print("iamge tensor",image_tensor)

  if color:   
    out = tf.reshape(image_tensor, [img_size, img_size, 3])
    out = tf.image.resize_images(out,resize_size,resize_size)
  else:
    # print("iamge",image_tensor)
    out = tf.reshape(image_tensor, [img_size, img_size, 1])
    out = tf.image.resize_images(out,resize_size,resize_size)
  if augment:
    out = tf.image.random_flip_left_right(out, seed=seed)
    # Add a random translation of up to 'max_x_offset' pixels by first cropping width by 'max_x_offset' pixels
    # (randomly distributed left or right), then padding zeros from the left.
  if augment_translation:
    max_x_offset = 2
    out = tf.image.random_crop(out, [img_size, img_size - max_x_offset], seed=seed*2)
    out = tf.image.pad_to_bounding_box(out, 0, max_x_offset, img_size, img_size)
  if augment_color:
    out = tf.image.random_hue(out, 0.5, seed=seed*3)
    out = tf.image.random_saturation(out, 0.0, 1.5, seed=seed*4)
  if whiten :
    # Bring to range [-1, 1]
    out = tf.cast(out, tf.float32) * (2. / 255) - 1
  else :
    # Brightest value is set to 1, darkest value to -1, then scaled
    # max = tf.cast(tf.reduce_max(out), tf.float32)
    # min = tf.cast(tf.reduce_min(out), tf.float32)
    # out = (tf.cast(out, tf.float32) - min) * 2. / (max - min) - 1
    out = tf.cast(out>0,tf.float32)*2 - 1
  # else:
  #   raise Exception("No whitening specified, aborted")
  return out


def make_image_producer(files, epochs, name, img_size,resize_size, shuffle, whiten, color, filename_seed=233,
                        augment=True, capacity=256, augment_color=False):
  with tf.variable_scope(name) as scope:
    gray_filename_queue = tf.train.string_input_producer(files, num_epochs=epochs, seed=filename_seed,
                                                         capacity=capacity, shuffle=shuffle)
    _, gray_files = tf.WholeFileReader(scope.name).read(gray_filename_queue)
    # print('gray_files', gray_files)
    channels = 3 if color else 1
    img = tf.image.decode_jpeg(gray_files, channels)
    # print("img", img)
    return preprocess(img, img_size,resize_size,
                         whiten=whiten, color=color, augment=augment, augment_color=augment_color)


def get_chair_images_and_sketches(epochs, img_size, depth_files, sketch_files,
                       shuffle=True, augment_color=False, filename_seed=233):
  img = make_image_producer(depth_files, epochs, 'rendered_producer', img_size,
                            shuffle, filename_seed=filename_seed, whiten='default', color=True, augment_color=augment_color)
  sketches = make_image_producer(sketch_files, epochs, 'sketch_producer', img_size,
                                 shuffle, filename_seed=filename_seed, whiten='sketch', color=False)
  return sketches, img


def get_chair_pipeline(batch_size, epochs, img_size, depth_files, sketch_files,
                       shuffle=True, augment_color=False, filename_seed=233):
  examples = get_chair_images_and_sketches(epochs, img_size, depth_files, sketch_files, shuffle=shuffle,
                                           augment_color=augment_color, filename_seed=filename_seed)
  return tf.train.batch(examples, batch_size=batch_size, num_threads=1, capacity=256 * 16)


def get_chair_pipeline_multi_thread(batch_size, epochs, img_size, depth_files, sketch_files,
                                    augment_color=False, min_queue_size=3000, read_threads=4):
  example_list = [get_chair_images_and_sketches(epochs, img_size, depth_files, sketch_files, shuffle=True,
                                                augment_color=augment_color, filename_seed=(i + 1) * 42)
                  for i in range(read_threads)]

  return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                     capacity=min_queue_size + batch_size * 16,
                                     min_after_dequeue=min_queue_size)


def get_chair_pipeline_training(batch_size, epochs):
  chair_folder = '/home/moser/shapenet_chairs_rendered2'
  sketch_folder = '/home/moser/shapenet_chairs_sketched2'
  img_size = 64
  size_suffix = str(img_size) + 'x' + str(img_size)
  return get_chair_pipeline(batch_size, epochs, img_size, get_rendered_files(chair_folder, size_suffix),
                            get_sketch_files(sketch_folder, size_suffix))

def get_chair_pipeline_training_recolor(batch_size, epochs):
  img_size = 64
  rendered_files = [x.strip() for x in open('recolor_experiment_shaded_images.txt').readlines()]
  sketched_files = [x.strip() for x in open('recolor_experiment_sketch_images.txt').readlines()]
  return get_chair_pipeline_multi_thread(batch_size, epochs, img_size, rendered_files,
                                         sketched_files, augment_color=True)


def read_tensor_record(filename_queue, img_size,resize_size, img_channels,sketch_channels):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={'image': tf.FixedLenFeature([], tf.string),
                'sketch': tf.FixedLenFeature([], tf.string)})
  image = tf.decode_raw(features['image'], tf.uint8)

  image.set_shape([img_size * img_size * img_channels])
  is_color_img = img_channels == 3
  is_color_sketch = sketch_channels==3
  # print("img size",array_ops.unpack(array_ops.shape(image)))
  image = preprocess(image, img_size,resize_size,
                     whiten=is_color_img, color=is_color_img, augment=False, augment_color=False)
  sketch = tf.decode_raw(features['sketch'], tf.uint8)
  sketch.set_shape([img_size * img_size * sketch_channels])
  sketch = preprocess(sketch, img_size,resize_size,
                      whiten=is_color_sketch, color=is_color_sketch, augment=False,augment_color=False)
  return sketch, image


def get_chair_pipeline_training_from_dump(dump_file, batch_size, epochs,
                                          image_size=256,resize_size=256, img_channels=3,sketch_channels=1, min_queue_size=2000, read_threads=4):
  with tf.variable_scope('dump_reader'):
    with tf.device('/cpu:0'):
      all_files = glob.glob(dump_file + '*')
      filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs)

      example_list = [read_tensor_record(filename_queue, image_size,resize_size, img_channels,sketch_channels)
                  for _ in range(read_threads)]

      return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=min_queue_size + batch_size * 16,
                                         min_after_dequeue=min_queue_size)
