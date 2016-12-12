from __future__ import print_function

import tensorflow as tf

from alexnet_v2 import alexnet_v2, alexnet_v2_arg_scope
from datasets import imagenet
from preprocess import preprocess_image

slim = tf.contrib.slim

# Setup
BATCH_SIZE = 256
DATA_DIR = '/data/cvg/imagenet/imagenet_tfrecords/'  # Directory of tf-records
MODEL_PATH = '/data/cvg/simon/data/logs/alex_net_run2/'
LOG_PATH = '/data/cvg/simon/data/logs/alex_net_eval/'
IM_SHAPE = [224, 224, 3]

print('Evaluating model: {}'.format(MODEL_PATH))

sess = tf.Session()
tf.logging.set_verbosity(tf.logging.DEBUG)

g = tf.Graph()
with sess.as_default():
    with g.as_default():
        global_step = slim.create_global_step()

        with tf.device('/cpu:0'):
            # Get test-data
            test_set = imagenet.get_split('test', dataset_dir=DATA_DIR)
            provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=4)
            [img_test, label] = provider.get(['image', 'label'])

            # Pre-process images
            img_test = preprocess_image(img_test, is_training=False, output_height=IM_SHAPE[0],
                                        output_width=IM_SHAPE[1])
            # Make batches
            imgs_test, labels_test = tf.train.batch([img_test, label], batch_size=BATCH_SIZE, num_threads=8,
                                                    capacity=8 * BATCH_SIZE)

        # Create the model
        with slim.arg_scope(alexnet_v2_arg_scope()):
            predictions = alexnet_v2(imgs_test, is_training=False)

        # Compute predicted label for accuracy
        preds_test = tf.argmax(predictions, 1)

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

        # Run the evaluation
        num_eval_steps = int(imagenet.SPLITS_TO_SIZES['test'] / BATCH_SIZE)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=1,
                                        eval_op=names_to_updates.values(),
                                        summary_op=tf.merge_summary(summary_ops))
