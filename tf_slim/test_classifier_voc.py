from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from ToonNet_AlexV2 import VAEGAN
from constants import LOG_DIR
from datasets import voc
from preprocess import preprocess_voc
from utils import montage_tf

slim = tf.contrib.slim

# Setup
finetuned = True
net_type = 'discriminator'
data = voc
model = VAEGAN(num_layers=5, batch_size=1)
TARGET_SHAPE = [224, 224, 3]
NUM_CONV_TRAIN = 5
TRAIN_SET = 'trainval'
TEST_SET = 'test'

if finetuned:
    MODEL_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_new_{}/'.format(
        data.NAME, model.name, net_type, NUM_CONV_TRAIN, TRAIN_SET))
    LOG_PATH = os.path.join(LOG_DIR, '{}_{}_finetune_{}_Retrain{}_new_{}/'.format(
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
            # Get test-data
            test_set = data.get_split(TEST_SET)
            test_provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=1, shuffle=False)
            [img_test, label_test] = test_provider.get(['image', 'label'])
            labels_test = tf.tile(tf.expand_dims(label_test, axis=0), [10, 1])
            imgs_test_t = tf.tile(tf.expand_dims(img_test, axis=0), [10, 1, 1, 1])
            imgs_test_p = tf.unstack(imgs_test_t, axis=0, num=10)
            imgs_test_p = [preprocess_voc(im, TARGET_SHAPE[0], TARGET_SHAPE[1], augment_color=False) for im in imgs_test_p]
            imgs_test = tf.stack(imgs_test_p, axis=0)

        # Get predictions
        preds_test = model.build_classifier(imgs_test, data.NUM_CLASSES, training=False)
        # preds_test = model.classifier(imgs_test, None, data.NUM_CLASSES, training=False, fine_tune=finetuned,
        #                               type=net_type)
        preds_test = tf.nn.sigmoid(preds_test)
        preds_test = tf.reduce_mean(preds_test, axis=0, keep_dims=True)

        summary_ops = []
        update_ops = []
        thresholds = [0.01 * i for i in range(101)]

        map_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
        for c in range(20):
            class_pred_test = tf.slice(preds_test, [0, c], size=[model.batch_size, 1])
            class_label_test = tf.slice(labels_test, [0, c], size=[model.batch_size, 1])

            # Choose the metrics to compute:
            prec_test, update_prec_test = slim.metrics.streaming_precision_at_thresholds(
                class_pred_test, class_label_test, thresholds)
            rec_test, update_rec_test = slim.metrics.streaming_recall_at_thresholds(
                class_pred_test, class_label_test, thresholds)

            ap_test = tf.Variable(0, dtype=tf.float32, collections=[ops.GraphKeys.LOCAL_VARIABLES])
            for i in range(11):
                ap_test += tf.reduce_max(prec_test * tf.cast(tf.greater_equal(rec_test, 0.1 * i), tf.float32)) / 11

            map_test += ap_test / 20

            op = tf.summary.scalar('ap_test_{}'.format(c), ap_test)
            op = tf.Print(op, [ap_test], 'ap_test_{}'.format(c), summarize=30)
            summary_ops.append(op)
            update_ops.append([update_prec_test, update_rec_test])

        op = tf.summary.scalar('map_test', map_test)
        op = tf.Print(op, [map_test], 'map_test', summarize=30)
        summary_ops.append(op)
        summary_ops.append(tf.summary.image('images/test', montage_tf(imgs_test, 3, 3), max_images=1))

        num_eval_steps = int(data.SPLITS_TO_SIZES[TEST_SET] / model.batch_size)
        slim.evaluation.evaluation_loop('', MODEL_PATH, LOG_PATH,
                                        num_evals=num_eval_steps,
                                        max_number_of_evaluations=100,
                                        eval_op=update_ops,
                                        summary_op=tf.summary.merge(summary_ops))
