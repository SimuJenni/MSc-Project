from keras.models import Model
from keras.utils.visualize_util import plot
from ToonNet import Generator

# Load a model
model = Generator(input_shape=(192, 192, 3), load_weights=False, w_outter=True)

# Plot the model
plot(model, to_file='model.png', show_shapes=True)
