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
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("train_steps", 100000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 24, "Training batch size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
                               "will use the worker hosts via their GRPC URLs (one client process "
                               "per worker host). Otherwise, will create an in-process TensorFlow "
                               "server.")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Get the data-set object
    data = Imagenet()

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    print("workers = %s" % worker_spec)
    print("num workers = %s" % num_workers)
    print("num gpus = %s" % FLAGS.num_gpus)

    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})

    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    is_chief = (FLAGS.task_index == 0)

    if FLAGS.num_gpus > 0:
        if FLAGS.num_gpus < num_workers:
            raise ValueError("number of gpus is less than number of workers")
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # set Keras learning phase to train
        keras.backend.set_learning_phase(1)
        # do not initialize variables on the fly
        keras.backend.manual_variable_initialization(True)

        # Build Keras model
        model, _, decoded = ToonNet(input_shape=data.get_dims(), batch_size=FLAGS.batch_size, out_activation='sigmoid',
                                    num_res_layers=10)

        # keras model predictions
        preds = model.outputs[0]

        # placeholder for training targets
        im_height, im_width, im_chan = data.get_dims()
        targets = tf.placeholder(tf.float32, shape=[None, im_height, im_width, im_chan])

        # Reconstruciton loss objective
        recon_loss = tf.reduce_mean(
            keras.objectives.mean_absolute_error(targets, preds))

        # apply regularizers if any
        if model.regularizers:
            total_loss = recon_loss * 1.  # copy tensor
            for regularizer in model.regularizers:
                total_loss = regularizer(total_loss)
        else:
            total_loss = recon_loss

        # set up TF optimizer
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                replica_id=FLAGS.task_index,
                name="toon_sync_replicas")

        train_step = opt.minimize(total_loss, global_step=global_step)

        if FLAGS.sync_replicas and is_chief:
            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op()

        init_op = tf.initialize_all_variables()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)
            sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(server.target,
                                                  config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op
            print("Starting chief queue runner and running init_tokens_op")
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        # Perform training
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        batch_gen = data.train_batch_generator(batch_size=FLAGS.batch_size)

        local_step = 0
        while not sv.should_stop():

            try:
                (train_data_batch, train_labels_batch) = batch_gen.next()
            except StopIteration:
                break

            _, step = sess.run([train_step, global_step],
                                           feed_dict={model.inputs[0]: train_data_batch,
                                                      targets: train_labels_batch})
            local_step += 1

            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                  (now, FLAGS.task_index, local_step, step))

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # # Validation feed
        # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        # print("After %d training step(s), validation cross entropy = %g" %
        #       (FLAGS.train_steps, val_xent))


        # Ask for all the services to stop.
        sv.stop()
        model.save('ToonNet_imagenet.h5')


if __name__ == "__main__":
    tf.app.run()
