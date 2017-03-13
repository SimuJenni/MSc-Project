from __future__ import print_function
import os

import tensorflow as tf

from ToonNet_VGG import VAEGAN
from constants import LOG_DIR
from datasets import stl10
from preprocess import preprocess_finetune_test, preprocess_finetune_test_edge

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'generator'
data = stl10
model = VAEGAN(num_layers=4, batch_size=500)
TARGET_SHAPE = [96, 96, 3]
RESIZE_SIZE = 96
NUM_CONV_TRAIN = 0
use_test_set = True

if finetuned:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_exp3_{}/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, 'train'))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_exp3_{}/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, 'train'))
else:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))

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
            num_eval_steps = int(data.SPLITS_TO_SIZES['test'] / model.batch_size)

            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test, edge_test] = provider.get(['image', 'label', 'edges'])

            # Pre-process data
            img_test = preprocess_finetune_test(img_test,
                                                output_height=TARGET_SHAPE[0],
                                                output_width=TARGET_SHAPE[1])
            edge_test = preprocess_finetune_test_edge(edge_test,
                                                output_height=TARGET_SHAPE[0],
                                                output_width=TARGET_SHAPE[1])
            # img_train, edge_train, _ = preprocess_toon_test(img_test, edge_test, img_test,
            #                                                  output_height=TARGET_SHAPE[0],
            #                                                  output_width=TARGET_SHAPE[1],
            #                                                 resize_side=RESIZE_SIZE)
            # Make batches
            imgs_test, labels_test, edges_test = tf.train.batch(
                [img_test, label_test, edge_test],
                batch_size=model.batch_size, num_threads=1)

        # Get predictions
        preds_test = model.classifier(imgs_test, edges_test, data.NUM_CLASSES, training=False,
                                      fine_tune=finetuned, type=net_type)

        # Compute predicted label for accuracy
        preds_test = tf.argmax(preds_test, 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_test),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
