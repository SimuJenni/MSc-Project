from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Invalid usage\n'
          'usage: preprocess_tiny-imagenet_train_data.py '
          '<train data dir>')
    sys.exit(-1)
  data_dir = sys.argv[1]

  dirs = os.listdir(data_dir)
  for d in dirs:
    target_dir = os.path.join(data_dir, d)
    im_dir = os.path.join(target_dir, 'images/')
    if os.path.exists(im_dir):
      images = os.listdir(im_dir)
      for im in images:
        old_filename = os.path.join(im_dir, im)
        new_filename = os.path.join(target_dir, im)
        os.rename(old_filename, new_filename)