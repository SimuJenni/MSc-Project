import gc
import tempfile
import time

import keras
import tensorflow as tf

from ToonNet import ToonNet
from datasets.Imagenet import Imagenet

# Flags for defining the tf.train.ClusterSpec
flags = tf.app.flags
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
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
    learning_rate = 0.001
    training_epochs = 1
    logs_path = "/tmp/ToonNet/"

    # Get the data-set object
    data = Imagenet()

    # Construct the cluster
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

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

            # set Keras learning phase to train
            keras.backend.set_learning_phase(1)

            # do not initialize variables on the fly
            keras.backend.manual_variable_initialization(True)

            # build Keras model
            model, _, decoded = ToonNet(input_shape=data.get_dims(),
                                        batch_size=batch_size,
                                        out_activation='sigmoid',
                                        num_res_layers=10,
                                        merge_mode='sum')

            # keras model predictions
            preds = model.output

            # placeholder for training targets
            im_height, im_width, im_chan = data.get_dims()
            targets = tf.placeholder(tf.float32, shape=[None, im_height, im_width, im_chan])

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

            with tf.control_dependencies(model.updates):
                barrier = tf.no_op(name='update_barrier')

            # define gradient updates
            with tf.control_dependencies([barrier]):
                train_step = optimizer.minimize(total_loss, global_step=global_step)

            # create a summary for our cost
            tf.scalar_summary("cost", total_loss)

            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 summary_op=summary_op,
                                 logdir=logs_path,
                                 init_op=init_op,
                                 recovery_wait_secs=1,
                                 global_step=global_step)

        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False,
                                     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        sess = sv.prepare_or_wait_for_session(server.target,
                                              config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)
        start_time = time_begin

        local_step = 0
        for epoch in range(training_epochs):
            print("Epoch {} / {}".format(epoch + 1, training_epochs))

            for X_train, Y_train in data.generator_train(batch_size):
                num_data = X_train.shape[0]

                for start in range(0, num_data, batch_size):
                    X_batch = X_train[start:(start + batch_size)]
                    Y_batch = Y_train[start:(start + batch_size)]
                    print(X_batch.shape, Y_batch.shape)
                    feed_dict = {model.inputs[0]: X_batch,
                                 targets: Y_batch}
                    _, step, train_loss = sess.run([train_step, global_step, total_loss],
                                                   feed_dict=feed_dict)
                    local_step += 1
                del X_train, Y_train
                gc.collect()
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print("Step: %d," % (local_step + 1),
                      " Epoch: %2d," % (epoch + 1),
                      " Cost: %.4f," % train_loss,
                      " Elapsed Time: %d" % elapsed_time)

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # Ask for all the services to stop.
        sv.stop()
        model.save('ToonNetDist_imagenet.h5')


if __name__ == "__main__":
    tf.app.run()
