import os
import time
import tensorflow as tf

from ops import *
from input_pipeline_rendered_data_vgg import get_chair_pipeline_training_from_dump

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=256, sample_size = 224, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.image_shape = image_shape
        self.image_size = image_shape[0]

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z = None

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.cg_dim = cg_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(is_train, name='d_bn1')
        self.d_bn2 = batch_norm(is_train, name='d_bn2')
        self.d_bn3 = batch_norm(is_train, name='d_bn3')

        self.g_bn0 = batch_norm(is_train, name='g_bn0')
        self.g_bn1 = batch_norm(is_train, name='g_bn1')
        self.g_bn2 = batch_norm(is_train, name='g_bn2')
        self.g_bn3 = batch_norm(is_train, name='g_bn3')

        self.g_s_bn1 = batch_norm(is_train, name='g_s_bn1')
        self.g_s_bn2 = batch_norm(is_train, name='g_s_bn2')
        self.g_s_bn3 = batch_norm(is_train, name='g_s_bn3')
        self.g_s_bn4 = batch_norm(is_train, convolutional=False, name='g_s_bn4')

        self.build_model(is_train)


    def build_model(self, is_train):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        
        self.abstract_size = self.sample_size // 2 ** 4
        sketches, images = get_chair_pipeline_training_from_dump('edge_256x256/edge_image_imagenet_256x256.tfrecords',
                                                                 self.batch_size,
                                                                 40, image_size=self.image_size,resize_size=self.sample_size,
                                                                 img_channels=self.c_dim,sketch_channels=3)
        
        test_sketches, test_images = get_chair_pipeline_training_from_dump('sketch_256x256/sketch_256x256.tfrecords',
                                                                 self.batch_size,
                                                                 1000000, image_size=self.image_size,resize_size=self.sample_size,
                                                                 img_channels=self.c_dim,sketch_channels=3)
        # self.images = tf.image.rgb_to_grayscale(images)
        self.images = images
        self.test_images = test_images
        self.test_sketches = test_sketches
        
        if is_train:
            # self.sketches = tf.image.rgb_to_grayscale(sketches)
            self.sketches = sketches

            print("sketches ", sketches)
            if self.z_dim:
                self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1, maxval=1, dtype=tf.float32)
        else:
            self.abstract = tf.placeholder(tf.float32, [None, self.abstract_size, self.abstract_size,
                                                        self.df_dim * 8 + self.z_dim])
            self.sketches = tf.placeholder(tf.float32, [None, self.sample_size,
                                                        self.sample_size, 3])
            if self.z_dim:
                self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope('generator') as scope:
            self.G,self.rep = self.generator(self.sketches, self.z,gflag=True)
            if not is_train:
                self.G_abstract = self.generator(self.abstract, from_abstract_representation=True, gflag=True)
            scope.reuse_variables()
            self.test_G,_ = self.generator(self.test_sketches, self.z,gflag=True)
            # print("test_G",self.test_G)
            scope.reuse_variables()
            self.C3_image = self.generator(self.images, self.z,gflag=False)
            scope.reuse_variables()
            self.C3_ge = self.generator(self.G, self.z,gflag=False)


        with tf.variable_scope('discriminator') as scope:
            self.D = self.discriminator(self.images, self.sketches)

            self.D_ = self.discriminator(self.G, self.sketches, reuse=True)

            self.testD = self.discriminator(self.test_images, self.test_sketches,reuse=True)

            self.testD_ = self.discriminator(self.test_G, self.test_sketches, reuse=True)   

        with tf.variable_scope('discriminator_feature') as scope:
            self.D_feature = self.discriminator_feature(self.C3_image, self.sketches)

            self.D_feature_ = self.discriminator_feature(self.rep, self.sketches, reuse=True)

        with tf.variable_scope('discriminator_loss') as scope:
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
            self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
            self.d_loss = self.d_loss_real + self.d_loss_fake

            self.d_loss_feat_real = binary_cross_entropy_with_logits(tf.ones_like(self.D_feature), self.D_feature)
            self.d_loss_feat_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_feature_), self.D_feature_)
            self.d_loss_feat = self.d_loss_feat_real + self.d_loss_feat_fake

            self.dtest_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.testD), self.testD)
            self.dtest_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.testD_), self.testD_)

        with tf.variable_scope('generator_loss') as scope:
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)
            self.g_loss_feat = binary_cross_entropy_with_logits(tf.ones_like(self.D_feature_), self.D_feature_)

            self.gtest_loss = binary_cross_entropy_with_logits(tf.ones_like(self.testD_), self.testD_)

        with tf.variable_scope('comparator_loss') as scope:
            self.c_loss = tf.reduce_mean(tf.square(self.C3_image - self.C3_ge))
            self.ctest_loss = tf.reduce_mean(tf.square((self.testD) - (self.testD_)))

        with tf.variable_scope('L2') as scope:
            # gray_generated = tf.image.rgb_to_grayscale(self.G)
            # self.whitened_generated = normalize_batch_of_images(self.G)
            # # gray_gt = tf.image.rgb_to_grayscale(self.images)
            # self.whitened_gt = normalize_batch_of_images(self.images)
            self.l2_loss = tf.reduce_mean(tf.square(self.G - self.images))
            self.l2test_loss = tf.reduce_mean(tf.square(self.test_G - self.test_images))

        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_conv_vars = [var for var in t_vars if 'g_conv_' in var.name]

        self.saver = tf.train.Saver( self.g_conv_vars +
                                    batch_norm.shadow_variables,
                                    max_to_keep=0)



    def train(self, config, run_string="???"):
        """Train DCGAN"""

        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(10*self.d_loss+1*self.d_loss_feat, var_list=self.d_vars, global_step=global_step)

        # Learning rate of generator is gradually decreasing.
        self.g_lr = tf.train.exponential_decay(config.learning_rate,
                                               global_step=global_step,
                                               decay_steps=20000,
                                               decay_rate=0.5,
                                               staircase=True)
        g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1) \
                          .minimize(10*self.g_loss+self.c_loss+self.l2_loss+1*self.g_loss_feat, var_list=self.g_vars)


        # See that moving average is also updated with g_optim.
        with tf.control_dependencies([g_optim]):
            g_optim = tf.group(self.bn_assigners)

        tf.initialize_all_variables().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.make_summary_ops()
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(config.summary_dir, graph_def=self.sess.graph_def)

        try:
            # Training
            while not coord.should_stop():
                # Update D and G network
                tic = time.time()

                _, _, errD_fake, errD_real,errD_feat_fake,errD_feat_real, errG, errL2,errC = self.sess.run([d_optim, g_optim, self.d_loss_fake,
                                                                  self.d_loss_real,self.d_loss_feat_fake,self.d_loss_feat_real, self.g_loss, self.l2_loss,self.c_loss])
               
                errDtest_fake, errDtest_real, errGtest, errL2test,errCtest = self.sess.run([self.dtest_loss_fake,
                                                                  self.dtest_loss_real, self.gtest_loss, self.l2test_loss,self.ctest_loss])
                # Run g_optim more to make sure that d_loss does not go to zero (different from paper)
                MAX_ADDITIONAL_UPDATES = 5
                errD_fake_threshold = 1e-3
                additional_G_runs = 0
                while errD_fake < errD_fake_threshold and additional_G_runs < MAX_ADDITIONAL_UPDATES:
                    [errD_fake, _] = self.sess.run([self.d_loss_fake, g_optim])
                    additional_G_runs += 1

                toc = time.time()

                counter += 1
                duration = toc - tic
                # print("Run: %s, Step: [%4d] time: %5.1f, last iter: %1.2f (%1.4f e/s), d_loss: %.8f, g_loss: %.8f, l2_loss: %.8f, G+: %2d"
                #     % (run_string, counter, toc - start_time, duration, self.batch_size / duration,
                #        errD_fake+errD_real, errG, errL2, additional_G_runs))
                print("Run: %s, Step: [%4d] time: %5.1f, c_loss: %.8f,d_loss_F: %.8f,d_loss_R: %.8f,d_loss_featF: %.8f,d_loss_featR: %.8f,g_loss: %.8f, L2_loss: %.8f, addG:%d"
                    % (run_string, counter, toc - start_time, errC,errD_fake,errD_real,errD_feat_fake,errD_feat_real,errG,errL2,additional_G_runs))
                print("  ctest_loss: %.8f,dtest_loss_F: %.8f,dtest_loss_R: %.8f,gtest_loss: %.8f, L2test_loss: %.8f"
                    % ( errCtest,errDtest_fake,errDtest_real,errGtest,errL2test))

                if counter % 50 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 500) == 2:
                    samples, sample_images, sample_sketches = self.sess.run([self.G, self.images, self.sketches])
                    test_samples, test_images, test_sketches = self.sess.run([self.test_G, self.test_images, self.test_sketches])

                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]
                if np.mod(counter, 2000) == 100:
                    self.save(config.checkpoint_dir, counter)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

    def discriminator(self, image, sketches, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # concated = tf.concat(3, [image, sketches])
        self.d_h0 = lrelu(conv2d(image, self.df_dim, name='d_conv_h0'))
        h1 = lrelu(self.d_bn1(conv2d(self.d_h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_conv_h2')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_conv_h3')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_lin_h3')

        return tf.nn.sigmoid(h4)

    def discriminator_feature(self, image, sketches, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # concated = tf.concat(3, [image, sketches])
        self.d_h0 = lrelu(conv2d(image, self.df_dim, name='d_conv_sh0'))
        # h1 = lrelu(self.d_bn1(conv2d(self.d_h0, self.df_dim*2, name='d_sh1_conv')))
        # h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_sh2_conv')))
        # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h1 = linear(tf.reshape(self.d_h0, [self.batch_size, -1]), 1, 'd_lin_sh1')

        return tf.nn.sigmoid(h1)

    def generator(self, sketches_or_abstract_representations, z=None, y=None, from_abstract_representation=False,gflag=False,pred=False):
        if from_abstract_representation:
            # Used when feeding abstract representation directly, not deriving it from a sketch.
            used_abstract = sketches_or_abstract_representations
            tf.get_variable_scope().reuse_variables()
        else:
            self.s0 = lrelu(conv2d(sketches_or_abstract_representations, self.df_dim, name='g_conv_s0'))
            # c0 = self.s0
            self.s1 = lrelu(self.g_s_bn1(conv2d(self.s0, self.df_dim * 2, name='g_conv_s1')))
            # c1 = self.s1
            self.s2 = lrelu(self.g_s_bn2(conv2d(self.s1, self.df_dim * 4, name='g_conv_s2')))
            # c2 = self.s2
            self.abstract_representation = lrelu(self.g_s_bn3(conv2d(self.s2, self.df_dim * 8, name='g_conv_s3')))
            # Size after 4 convolutions with stride 2.
            if self.z_dim:
                z_slices = tf.mul(tf.ones([self.batch_size, self.abstract_size, self.abstract_size, self.z_dim]),
                                  tf.reshape(self.z, [self.batch_size, 1, 1, self.z_dim]))
                self.representation = tf.concat(3, [self.abstract_representation, z_slices])
            used_abstract = self.abstract_representation

            # if pred:
            #     #fc6
            #     #fc(4096, name='fc6')
            #     fc6 = tf.nn.relu_layer(tf.reshape(self.abstract_representation, [1, int(prod(self.abstract_representation.get_shape()[1:]))]), fc6W, fc6b,'')

            #     #fc7
            #     #fc(4096, name='fc7')
            #     fc7 = tf.nn.relu_layer(fc6)

            #     #fc8
            #     #fc(1000, relu=False, name='fc8')
            #     fc8 = tf.nn.xw_plus_b(fc7)

            #     #prob
            #     #softmax(name='prob'))
            #     prob = tf.nn.softmax(fc8)

            # c3 = self.abstract_representation

        if gflag:

            h1 = deconv2d(used_abstract, [self.batch_size, self.abstract_size * 2,
                                                         self.abstract_size * 2, self.gf_dim*4 + self.z_dim],name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, self.abstract_size * 4, self.abstract_size * 4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, self.abstract_size * 8, self.abstract_size * 8, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv2d(h3, [self.batch_size, self.abstract_size * 16, self.abstract_size * 16, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4),self.abstract_representation
        else:
            return self.abstract_representation


    def make_summary_ops(self):
        tf.image_summary('generator', self.G)
        tf.image_summary('sketch', self.sketches)
        tf.image_summary('images', self.images)
        tf.scalar_summary('d_loss_fake', self.d_loss_fake)
        tf.scalar_summary('d_loss_real', self.d_loss_real)
        tf.scalar_summary('d_loss_feat_fake', self.d_loss_feat_fake)
        tf.scalar_summary('d_loss_feat_real', self.d_loss_feat_real)
        tf.scalar_summary('g_loss', self.g_loss)
        tf.scalar_summary('g_loss_feat', self.g_loss_feat)
        tf.scalar_summary('d_loss', self.d_loss)
        tf.scalar_summary('g_lr', self.g_lr)
        tf.scalar_summary('l2_loss', self.l2_loss)
        tf.scalar_summary('c_loss', self.c_loss)
        tf.histogram_summary('abstract_representation', self.abstract_representation)
        tf.histogram_summary('d_h0', self.d_h0)
        if self.z_dim:
            with tf.variable_scope('z_stats') as scope:
                length = tf.sqrt(tf.reduce_sum(tf.square(self.z)))
                tf.scalar_summary('length_z', length)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name
