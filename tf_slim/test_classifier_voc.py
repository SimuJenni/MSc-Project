from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from ToonNet import VAEGAN
from constants import LOG_DIR
from datasets import voc
from preprocess import preprocess_voc
from utils import montage_tf

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'discriminator'
data = voc
model = VAEGAN(num_layers=5, batch_size=200)
TARGET_SHAPE = [128, 128, 3]
NUM_CONV_TRAIN = 5
TRAIN_SET = 'train'
TEST_SET = 'val'

if finetuned:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_final_{}_imnet/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, TRAIN_SET))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_final_{}_imnet/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, TRAIN_SET))
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
            # Get train-data
            train_set = data.get_split(TRAIN_SET)
            train_provider = slim.dataset_data_provider.DatasetDataProvider(train_set, num_readers=1, shuffle=False)
            [img_train, label_train] = train_provider.get(['image', 'label'])

            # Get test-data
            test_set = data.get_split(TEST_SET)
            test_provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test] = test_provider.get(['image', 'label'])

            # Pre-process data
            img_train = preprocess_voc(img_train, TARGET_SHAPE[0], TARGET_SHAPE[1])
            img_test = preprocess_voc(img_test, TARGET_SHAPE[0], TARGET_SHAPE[1], aspect_ratio_range=[1.0, 1.0],
                                      area_range=[1.0, 1.0], augment_color=False)

            # Make batches
            imgs_test, labels_test = tf.train.batch([img_test, label_test], batch_size=model.batch_size)
            imgs_train, labels_train = tf.train.batch([img_train, label_train], batch_size=model.batch_size)

        # Get predictions
        preds_test = model.classifier(imgs_test, None, data.NUM_CLASSES, training=False, fine_tune=finetuned,
                                      type=net_type, weight_decay=0.0001, bn_decay=0.99, keep_prob=0.5)
        preds_train = model.classifier(imgs_train, None, data.NUM_CLASSES, training=False, fine_tune=finetuned,
                                       type=net_type, reuse=True, weight_decay=0.0001, bn_decay=0.99, keep_prob=0.5)
        preds_test = tf.nn.sigmoid(preds_test)
        preds_train = tf.nn.sigmoid(preds_train)

        summary_ops = []
        update_ops = []
        thresholds = [0.01 * i for i in range(101)]

        map_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
        map_train = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
        for c in range(20):
            class_pred_train = tf.slice(preds_train, [0, c], size=[model.batch_size, 1])
            class_pred_test = tf.slice(preds_test, [0, c], size=[model.batch_size, 1])

            class_label_train = tf.slice(labels_train, [0, c], size=[model.batch_size, 1])
            class_label_test = tf.slice(labels_test, [0, c], size=[model.batch_size, 1])

            # Choose the metrics to compute:
            prec_train, update_prec_train = slim.metrics.streaming_precision_at_thresholds(
                class_pred_train, class_label_train, thresholds)
            prec_test, update_prec_test = slim.metrics.streaming_precision_at_thresholds(
                class_pred_test, class_label_test, thresholds)
            rec_train, update_rec_train = slim.metrics.streaming_recall_at_thresholds(
                class_pred_train, class_label_train, thresholds)
            rec_test, update_rec_test = slim.metrics.streaming_recall_at_thresholds(
                class_pred_test, class_label_test, thresholds)

            ap_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
            ap_train = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
            for i in range(11):
                ap_test += tf.reduce_max(prec_test * tf.cast(tf.greater(rec_test, 0.1 * i), tf.float32)) / 11
                ap_train += tf.reduce_max(prec_train * tf.cast(tf.greater(rec_train, 0.1 * i), tf.float32)) / 11

            map_test += ap_test/20
            map_train += ap_train/20

            op = tf.scalar_summary('ap_test_{}'.format(c), ap_test)
            op = tf.Print(op, [ap_test], 'ap_test_{}'.format(c), summarize=30)
            summary_ops.append(op)
            op = tf.scalar_summary('ap_train_{}'.format(c), ap_train)
            op = tf.Print(op, [ap_train], 'ap_train_{}'.format(c), summarize=30)
            summary_ops.append(op)
            update_ops.append([update_prec_train, update_prec_test, update_rec_train, update_rec_test])

        op = tf.scalar_summary('map_test', map_test)
        op = tf.Print(op, [map_test], 'map_test', summarize=30)
        summary_ops.append(op)
        op = tf.scalar_summary('map_train', map_train)
        op = tf.Print(op, [map_train], 'map_train', summarize=30)
        summary_ops.append(op)

        summary_ops.append(tf.image_summary('images/train', montage_tf(imgs_train, 3, 3), max_images=1))
        summary_ops.append(tf.image_summary('images/test', montage_tf(imgs_test, 3, 3), max_images=1))

        num_eval_steps = int(data.SPLITS_TO_SIZES[TEST_SET] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=20,
                                        eval_op=update_ops,
                                        summary_op=tf.merge_summary(summary_ops))
