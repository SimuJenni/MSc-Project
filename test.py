import numpy as np
import os
from keras.layers import Input
from keras.models import Model

from ToonDataGenerator import ImageDataGenerator
from datasets import TinyImagenetToon, CIFAR10_Toon
from utils import montage
from ToonNet import ToonGenTransp, ToonDisc

# data = CIFAR10()
# class_dirs = os.listdir(data.val_dir)
# y_true = []
# for i, c in enumerate(class_dirs):
#     y_true += [i]*len(os.listdir(os.path.join(data.val_dir, c)))
# print(y_true)
# print(len(y_true))
# print(data.num_val)
# datagen = ImageDataGenerator()
# X_test, Y_test = datagen.flow_from_directory(data.val_dir, batch_size=2, target_size=data.target_size).next()
#
# gan, gen_gan, disc_gan = GANwEncoder(data.dims, load_weights=False)
# h = gan.fit(x=X_test, y=[np.ones((len(Y_test), 1)), np.zeros((2,4,4,512)), Y_test], nb_epoch=1, batch_size=2, verbose=0)
# print(h.history.keys())
# print(h.history)
# print(h.history['ToonGenerator_nr8_loss'][0])
# print(gan.output_names)
# print(h.history['{}_loss'.format(gan.output_names[0])][0])
#
# print(h.history['%s_loss' % gan.output_names[0]])

# data = CIFAR10_Toon()
# datagen = ImageDataGenerator(rotation_range=10,
#         width_shift_range=0.05,
#         height_shift_range=0.05,
#         shear_range=0.05,
#         zoom_range=[0.9, 1.0],
#         horizontal_flip=True,
#         fill_mode='nearest')
#
# test_toon, test_edges, test_img = datagen.flow_from_directory(data.train_dir, batch_size=64, target_size=data.dims[:2]).next()
# montage(test_toon[:, :, :] * 0.5 + 0.5, 'Test-Toon.jpeg')
# montage(np.squeeze(test_edges[:, :, :]), 'Test-Edge.jpeg', gray=True)
# montage(test_img[:, :, :] * 0.5 + 0.5, 'Test-Image.jpeg')


input_gen = Input(batch_shape=(64, 32, 32, 3))
decoded, _ = ToonGenTransp(input_gen, num_layers=3, batch_size=64)
generator = Model(input_gen, decoded)
p_out, d_out = ToonDisc(input_gen, num_layers=3)
discriminator = Model(input_gen, [p_out, d_out])

# Compile
generator.compile(loss='mse', optimizer='adam')
generator.summary()
discriminator.compile(loss='mse', optimizer='adam')
discriminator.summary()

# gan, generator, discriminator = ToonGAN(input_shape=(32, 32, 3), num_layers=3, batch_size=128, train_disc=True)


