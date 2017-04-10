import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

import os
import sys
import numpy as np

from utils import montage_tf, get_variables_to_train, assign_from_checkpoint_fn
from constants import LOG_DIR

slim = tf.contrib.slim


class ToonNet_Tester:
    def __init__(self, model, dataset, pre_processor, tag='default'):
        tf.logging.set_verbosity(tf.logging.DEBUG)
        self.sess = tf.Session()
        self.graph = tf.Graph()
        self.model = model
        self.dataset = dataset
        self.tag = tag
        self.additional_info = None
        self.im_per_smry = 4
        self.summaries = {}
        self.pre_processor = pre_processor
        self.num_eval_steps = None
        with self.sess.as_default():
            with self.graph.as_default():
                self.global_step = slim.create_global_step()

    def get_save_dir(self):
        fname = '{}_{}_{}_finetune'.format(self.dataset.name, self.model.name, self.tag)
        if self.additional_info:
            fname = '{}_{}'.format(fname, self.additional_info)
        return os.path.join(LOG_DIR, '{}/'.format(fname))

    def get_test_batch(self, dataset_id):
        # Get the training dataset
        if dataset_id:
            test_set = self.dataset.get_split(dataset_id)
            self.num_eval_steps = (self.dataset.get_num_dataset(dataset_id) / self.model.batch_size)
        else:
            test_set = self.dataset.get_trainset()
            self.num_eval_steps = (self.dataset.get_num_train() / self.model.batch_size)
        print('Number of evaluation steps: {}'.format(self.num_eval_steps))
        provider = slim.dataset_data_provider.DatasetDataProvider(test_set, num_readers=2,
                                                                  common_queue_capacity=4 * self.model.batch_size,
                                                                  common_queue_min=self.model.batch_size)
        images_and_labels = []
        for thread_id in range(8):
            # Parse a serialized Example proto to extract the image and metadata.
            [img_test, label_test] = provider.get(['image', 'label'])
            label_test -= self.dataset.label_offset

            # Pre-process data
            img_test = self.pre_processor.process_transfer_train(img_test, thread_id)
            images_and_labels.append([img_test, label_test])

        # Make batches
        imgs_test, labels_test = tf.train.batch_join(
            images_and_labels,
            batch_size=self.model.batch_size,
            capacity=4 * self.model.batch_size)

        return imgs_test, labels_test

    def make_image_summaries(self, edges_train, img_gen, img_rec, imgs_train, toons_train):
        tf.image_summary('imgs/generator out', montage_tf(img_gen, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/autoencoder', montage_tf(img_rec, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/ground truth', montage_tf(imgs_train, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/cartoons', montage_tf(toons_train, 1, self.im_per_smry), max_images=1)
        tf.image_summary('imgs/edge maps', montage_tf(edges_train, 1, self.im_per_smry), max_images=1)

    def test_classifier(self, model_path, num_conv_trained=None, dataset_id=None):
        print('Restoring from: {}'.format(model_path))
        self.additional_info = 'conv_{}'.format(num_conv_trained)
        with self.sess.as_default():
            with self.graph.as_default():
                # Get training batches
                imgs_test, labels_test = self.get_test_batch(dataset_id)

                # Get predictions
                predictions = self.model.build_classifier(imgs_test, self.dataset.num_classes, training=False)

                # Compute predicted label for accuracy
                preds_test = tf.argmax(predictions, 1)

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

                slim.evaluation.evaluation_loop('', model_path, self.get_save_dir(),
                                                num_evals=self.num_eval_steps,
                                                max_number_of_evaluations=1,
                                                eval_op=names_to_updates.values(),
                                                summary_op=tf.merge_summary(summary_ops))
