from __future__ import print_function
import time
import sys

import keras
import tensorflow as tf
from keras import backend as K
from constants import LOG_DIR

from ToonNet import ToonNet
from datasets.Imagenet import Imagenet
from DataGenerator import ImageDataGenerator

# Flags for defining the tf.train.ClusterSpec
flags = tf.app.flags
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer('gpu_device_id', -1, 'gpu device id')
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # config
    batch_size = 32
    learning_rate = 0.0005
    training_epochs = 1
    logs_path = LOG_DIR

    # Get the data-set object
    data = Imagenet()
    datagen = ImageDataGenerator()

    # Construct the cluster
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    num_replicas_to_aggregate = len(worker_spec)

    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    is_chief = (FLAGS.task_index == 0)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, FLAGS.gpu_device_id)

        # Between-graph replication
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                      cluster=cluster)):
            # count the number of updates
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # do not initialize variables on the fly
            K.manual_variable_initialization(True)

            # build Keras model
            model, _, decoded = ToonNet(input_shape=data.dims,
                                        batch_size=batch_size,
                                        out_activation='sigmoid',
                                        num_res_layers=10,
                                        merge_mode='sum')

            # keras model predictions
            preds = model.output

            # placeholder for training targets
            targets = tf.placeholder(tf.float32, shape=(None, ) + data.dims, name='targets')

            # reconstruction loss objective
            recon_loss = tf.reduce_mean(keras.objectives.mean_absolute_error(targets, preds))

            # apply regularizers if any
            if model.regularizers:
                total_loss = recon_loss * 1.  # copy tensor
                for regularizer in model.regularizers:
                    total_loss = regularizer(total_loss)
            else:
                total_loss = recon_loss

            # set up TF optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Create synchronous replica optimizer.
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer,
                replicas_to_aggregate=num_replicas_to_aggregate,
                replica_id=FLAGS.task_index,
                total_num_replicas=num_replicas_to_aggregate)

            # Batchnorm updates
            with tf.control_dependencies(model.updates):
                total_loss = tf.identity(total_loss)

            # Compute gradients with respect to the loss.
            grads = optimizer.compute_gradients(total_loss)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            apply_gradients_op = optimizer.apply_gradients(grads, global_step=global_step)

            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')

            # create a summary for our cost
            tf.scalar_summary("cost", total_loss)

            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 saver=saver,
                                 summary_op=summary_op,
                                 logdir=logs_path,
                                 init_op=init_op,
                                 global_step=global_step,
                                 save_model_secs=600)

        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False,
                                     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
            K.set_session(sess)

            # Perform training
            time_begin = time.time()
            print("Training begins @ %f" % time_begin)
            start_time = time_begin

            local_step = 0
            for epoch in range(training_epochs):
                print("Epoch {} / {}".format(epoch + 1, training_epochs))

                for X_batch, Y_batch in datagen.flow_from_directory(data.train_dir, batch_size=batch_size):
                    feed_dict = {model.inputs[0]: X_batch,
                                 targets: Y_batch,
                                 K.learning_phase(): 1}
                    _, step, train_loss = sess.run([train_op, global_step, total_loss],
                                                   feed_dict=feed_dict)
                    local_step += 1
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Global-Step: %d," % step,
                          " Local-Step: %d" % local_step,
                          " Epoch: %2d," % (epoch + 1),
                          " Cost: %.4f," % train_loss,
                          " Elapsed Time: %d" % elapsed_time)
                    sys.stdout.flush()

            time_end = time.time()
            print("Training ends @ %f" % time_end)
            training_time = time_end - time_begin
            print("Training elapsed time: %f s" % training_time)

            # Ask for all the services to stop.
            sv.stop()
            model.save('ToonNetDist_imagenet.h5')


if __name__ == "__main__":
    tf.app.run()
