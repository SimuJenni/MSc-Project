from __future__ import print_function

import os

import tensorflow as tf

from ToonNet_AlexV2 import VAEGAN
from constants import LOG_DIR, IMAGENET_TF_DATADIR, IMAGENET_TF_256_DATADIR
from datasets import imagenet
from preprocess import preprocess_finetune_test, preprocess_imagenet_256_test, preprocess_imagenet_musub_test

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'discriminator'
data = imagenet
model = VAEGAN(num_layers=5, batch_size=500)
TARGET_SHAPE = [224, 224, 3]
RESIZE_SIZE = 224
NUM_CONV_TRAIN = 0

if finetuned:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain_final/'.format(data.NAME, model.name,
                                                                                    NUM_CONV_TRAIN))
    LOG_PATH = MODEL_PATH
else:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_classifier/'.format(data.NAME, model.name))
    LOG_PATH = MODEL_PATH

print('Evaluating model: {}'.format(MODEL_PATH))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        with tf.device('/cpu:0'):
            # Get test-data
            test_set = data.get_split('validation', dataset_dir=IMAGENET_TF_256_DATADIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test] = provider.get(['image', 'label'])
            label_test -= data.LABEL_OFFSET

            # Pre-process data
            img_test = preprocess_imagenet_256_test(img_test, output_height=TARGET_SHAPE[0],
                                                    output_width=TARGET_SHAPE[1])

            # Make batches
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=model.batch_size,
                                                    num_threads=1)

        # Get predictions
        preds_test = model.build_classifier(imgs_test, data.NUM_CLASSES, training=False)

        # Compute predicted label for accuracy
        preds_test = tf.argmax(tf.nn.softmax(preds_test), 1)

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(preds_test, labels_test),
        })

        # Create the summary ops such that they also print out to std output:
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        num_eval_steps = int(data.SPLITS_TO_SIZES['validation'] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.summary.merge(summary_ops))
