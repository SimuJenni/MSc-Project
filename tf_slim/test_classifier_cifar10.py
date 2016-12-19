from __future__ import print_function

import os

import tensorflow as tf

from ToonNetCondVAEGAN import AEGAN
from constants import LOG_DIR
from datasets import cifar10
from preprocess import preprocess_toon_test

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'discriminator'
data = cifar10
model = AEGAN(num_layers=4, batch_size=500, data_size=data.SPLITS_TO_SIZES['train'])
TARGET_SHAPE = [32, 32, 3]
RESIZE_SIZE = max(TARGET_SHAPE[0], data.MIN_SIZE)

if finetuned:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_new_settings2/'.format(data.NAME, model.name, net_type))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_eval_new_settings2/'.format(data.NAME, model.name, net_type))
else:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_classifier_eval/'.format(data.NAME, model.name))

print('Evaluating model: {}'.format(MODEL_PATH))

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
            [img_test, edge_test, toon_test, label_test] = provider.get(['image', 'edges', 'cartoon', 'label'])

            # Pre-process data
            img_test, edge_test, toon_test = preprocess_toon_test(img_test, edge_test, toon_test,
                                                                  output_height=TARGET_SHAPE[0],
                                                                  output_width=TARGET_SHAPE[1],
                                                                  resize_side=RESIZE_SIZE)
            # Make batches
            imgs_test, edges_test, toons_test, labels_test = tf.train.batch(
                [img_test, edge_test, toon_test, label_test],
                batch_size=model.batch_size, num_threads=4)

        # Get predictions
        preds_test = model.classifier(imgs_test, edges_test, toons_test, data.NUM_CLASSES, training=False,
                                      fine_tune=finetuned, type=net_type)

        # Compute predicted label for accuracy
        preds_test = tf.argmax(preds_test, 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_test),
            'precision': slim.metrics.streaming_precision(preds_test, labels_test),
            'recall': slim.metrics.streaming_recall(preds_test, labels_test),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_eval_steps = int(data.SPLITS_TO_SIZES['test'] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
