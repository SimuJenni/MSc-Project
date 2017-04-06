import tensorflow as tf

from ToonNet import ToonNet

slim = tf.contrib.slim

model = ToonNet(num_layers=5, batch_size=128)

x = tf.Variable(tf.random_normal([1, 128, 128, 3], stddev=2), name='x')
net = model.discriminator.discriminate(x, training=False, with_fc=False)

print('Trainable variables: {}'.format([v.op.name for v in slim.get_variables_to_restore()]))
