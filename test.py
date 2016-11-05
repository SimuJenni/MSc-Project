import numpy as np
import os

from ToonDataGenerator import ImageDataGenerator
from datasets import TinyImagenetToon, CIFAR10_Toon
from utils import montage

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

data = CIFAR10_Toon()
datagen = ImageDataGenerator(rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=[0.9, 1.0],
        horizontal_flip=True,
        fill_mode='nearest')

test_X, test_Y = datagen.flow_from_directory(data.train_dir, batch_size=64, target_size=data.dims[:2]).next()
montage(test_X[:, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
montage(test_Y[:, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')
