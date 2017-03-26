from __future__ import print_function
import os

import tensorflow as tf
from ToonNet_AlexV2 import VAEGAN
from constants import LOG_DIR, IMAGENET_TF_256_DATADIR
from datasets import imagenet
from preprocess import preprocess_imagenet_256_test

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'discriminator'
data = imagenet
model = VAEGAN(num_layers=5, batch_size=1)
TARGET_SHAPE = [224, 224, 3]
NUM_CONV_TRAIN = 0
TEST_SET = 'validation'
RESIZE_SIZE = 224

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

            labels_test = tf.tile(tf.expand_dims(label_test, dim=0), [10, 1])
            imgs_test_t = tf.tile(tf.expand_dims(img_test, dim=0), [10, 1, 1, 1])
            imgs_test_p = tf.unpack(imgs_test_t, axis=0, num=10)
            imgs_test_p = [preprocess_imagenet_256_test(im, TARGET_SHAPE[0], TARGET_SHAPE[1]) for im in imgs_test_p]
            imgs_test = tf.pack(imgs_test_p, axis=0)

        # Get predictions
        preds_test = model.build_classifier(imgs_test, data.NUM_CLASSES, training=False)
        preds_test = tf.nn.softmax(preds_test)
        preds_test = tf.reduce_mean(preds_test, reduction_indices=0, keep_dims=True)

        # Compute predicted label for accuracy
        preds_test = tf.argmax(tf.nn.softmax(preds_test), 1)

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

        num_eval_steps = int(data.SPLITS_TO_SIZES[TEST_SET] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
