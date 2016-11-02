import numpy as np

from ToonDataGenerator import ImageDataGenerator
from datasets import TinyImagenetToon, CIFAR10_Toon
from utils import montage

data = CIFAR10_Toon()
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



# test_X, test_Y = datagen.flow_from_directory(data.train_dir, batch_size=64, target_size=data.dims[:2],
#                                              xy_fun=Y2X_Y2Y).next()
# montage(test_X[:, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
# montage(test_Y[:, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')

# # Load a model
# model = Generator(input_shape=(192, 192, 3), load_weights=False, w_outter=True)
#
# # Plot the model
# plot(model, to_file='model.png', show_shapes=True)
