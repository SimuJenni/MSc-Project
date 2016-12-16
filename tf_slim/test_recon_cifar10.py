from __future__ import print_function

import os

import tensorflow as tf

from ToonNetVAEGAN import AEGAN
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_toon_test
from utils import montage

slim = tf.contrib.slim

# Setup
data = cifar10
model = AEGAN(num_layers=4, batch_size=500, data_size=data.SPLITS_TO_SIZES['train'])
TARGET_SHAPE = [32, 32, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)
MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_new_settings/'.format(data.NAME, model.name))
LOG_PATH = os.path.join(LOG_DIR, '{}_{}_new_settings_recon_test/'.format(data.NAME, model.name))

print('Testing model: {}'.format(MODEL_PATH))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        with tf.device('/cpu:0'):
            # Get test-data
            test_set = data.get_split('test')
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
            [img_test, edge_test, toon_test] = provider.get(['image', 'edges', 'cartoon'])

            # Pre-process data
            img_test, edge_test, toon_test = preprocess_toon_test(img_test, edge_test, toon_test,
                                                                  output_height=TARGET_SHAPE[0],
                                                                  output_width=TARGET_SHAPE[1],
                                                                  resize_side=RESIZE_SIZE)
            # Make batches
            imgs_test, edges_test, toons_test = tf.train.batch(
                [img_test, edge_test, toon_test],
                batch_size=model.batch_size, num_threads=4)

        # Create the model
        img_rec, gen_rec, disc_out, enc_dist, gen_dist, enc_mu, gen_mu, enc_logvar, gen_logvar = \
            model.net(imgs_test, toons_test, edges_test)
        preds_test = tf.argmax(disc_out, 1)

        # Get labels for discriminator
        labels_disc = model.disc_labels()

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_disc),
            'MSE-ae': slim.metrics.streaming_mean_squared_error(img_rec, imgs_test),
            'MSE-generator': slim.metrics.streaming_mean_squared_error(gen_rec, imgs_test),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.image_summary('images/generator', montage(gen_rec, 8, 8), max_images=1))
        summary_ops.append(tf.image_summary('images/ae', montage(img_rec, 8, 8), max_images=1))
        summary_ops.append(tf.image_summary('images/ground-truth', montage(imgs_test, 8, 8), max_images=1))
        summary_ops.append(tf.image_summary('images/cartoons', montage(toons_test, 8, 8), max_images=1))
        summary_ops.append(tf.image_summary('images/edges', montage(edges_test, 8, 8), max_images=1))

        num_eval_steps = int(data.SPLITS_TO_SIZES['test'] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
