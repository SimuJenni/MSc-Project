from keras.utils.visualize_util import plot
from ToonNet import Generator
from datasets import TinyImagenet
from DataGenerator import ImageDataGenerator, X2X_X2Y
from utils import montage

data = TinyImagenet()
datagen = ImageDataGenerator(horizontal_flip=True)
test_X, test_Y = datagen.flow_from_directory(data.train_dir, xy_fun=X2X_X2Y, batch_size=64).next()
montage(test_X[:, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
montage(test_Y[:, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')


# # Load a model
# model = Generator(input_shape=(192, 192, 3), load_weights=False, w_outter=True)
#
# # Plot the model
# plot(model, to_file='model.png', show_shapes=True)
