from keras.layers import Input, Dense, BatchNormalization, merge
from keras.models import Model
from keras.utils.visualize_util import plot

# this returns a tensor
in1 = Input(shape=(10,))
in2 = Input(shape=(10,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(in1)
bn = BatchNormalization()(x)
x = Dense(64, activation='relu')(merge([bn, x], mode='concat'))
predictions = Dense(10, activation='softmax')(x)

# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=in1, output=predictions)
plot(model, to_file='model.png', show_shapes=True)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])