from __future__ import print_function

import os
import sys

import tensorflow as tf

from ToonNet_Alex_comp_centerBN_relu import VAEGAN
from constants import LOG_DIR
from datasets import imagenet
from preprocess import preprocess_toon_test
from utils import montage_tf, weights_montage
from constants import IMAGENET_SMALL_TF_DATADIR

slim = tf.contrib.slim

# Setup
data = imagenet
model = VAEGAN(num_layers=5, batch_size=128)
TARGET_SHAPE = [96, 96, 3]
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_final/'.format(data.NAME, model.name))
LOG_PATH = os.path.join(LOG_DIR, '{}_{}_final_recon_test/'.format(data.NAME, model.name))

print('Testing model: {}'.format(MODEL_PATH))
sys.stdout.flush()

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        with tf.device('/cpu:0'):
            # Get test-data
            test_set = data.get_split('validation', dataset_dir=IMAGENET_SMALL_TF_DATADIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, edge_test, toon_test] = provider.get(['image', 'edges', 'cartoon'])

            # Pre-process data
            img_test, edge_test, toon_test = preprocess_toon_test(img_test, edge_test, toon_test,
                                                                  output_height=TARGET_SHAPE[0],
                                                                  output_width=TARGET_SHAPE[1],
                                                                  resize_side=96)
            # Make batches
            imgs_test, edges_test, toons_test = tf.train.batch(
                [img_test, edge_test, toon_test],
                batch_size=model.batch_size, num_threads=1)

        # Create the model
        img_rec, gen_rec, disc_out, enc_dist, gen_dist, enc_mu, gen_mu, enc_logvar, gen_logvar = \
            model.net(imgs_test, toons_test, edges_test)
        preds_test = tf.argmax(disc_out, 1)

        # Get labels for discriminator
        labels_disc = tf.argmax(model.disc_labels(), 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'MSE-ae': slim.metrics.streaming_mean_squared_error(img_rec, imgs_test),
            'MSE-generator': slim.metrics.streaming_mean_squared_error(gen_rec, imgs_test),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.image_summary('images/generator', montage_tf(gen_rec[:100], 10, 10), max_images=1))
        summary_ops.append(tf.image_summary('images/ae', montage_tf(img_rec[:100], 10, 10), max_images=1))
        summary_ops.append(tf.image_summary('images/ground-truth', montage_tf(imgs_test[:100], 10, 10), max_images=1))
        summary_ops.append(tf.image_summary('images/cartoons', montage_tf(toons_test[:100], 10, 10), max_images=1))
        summary_ops.append(tf.image_summary('images/edges', montage_tf(edges_test[:100], 10, 10), max_images=1))

        with tf.variable_scope('discriminator', reuse=True):
            weights_disc_1 = slim.variable('conv_1/weights')
        summary_ops.append(tf.image_summary('images/weights_disc_1', weights_montage(weights_disc_1, 6, 16),
                                            max_images=1))

        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=1,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
