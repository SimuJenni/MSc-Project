import numpy as np
import os
from keras.layers import Input
from keras.models import Model

from ToonDataGenerator import ImageDataGenerator
from datasets import TinyImagenetToon, CIFAR10_Toon
from utils import montage
from ToonNet import ToonGen, ToonDisc

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
# test_X, test_Y = datagen.flow_from_directory(data.train_dir, batch_size=64, target_size=data.dims[:2]).next()
# montage(test_X[:, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
# montage(test_Y[:, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')


# input_gen = Input(batch_shape=(64, 32, 32, 3))
# decoded, _ = ToonGen(input_gen, num_layers=3, batch_size=64)
# generator = Model(input_gen, decoded)
# p_out, d_out, _ = ToonDisc(input_gen, num_layers=3)
# discriminator = Model(input_gen, [p_out, d_out])
#
# # Compile
# generator.compile(loss='mse', optimizer='adam')
# generator.summary()
# discriminator.compile(loss='mse', optimizer='adam')
# discriminator.summary()

from keras.layers import Input, merge, Lambda
from keras.models import Model
import numpy as np
import keras.backend as K

def cos_sim(x, y):
    def cos(x):
        y_true = K.l2_normalize(x[0], axis=-1)
        y_pred = K.l2_normalize(x[1], axis=-1)
        return K.batch_dot(y_true, y_pred, axes=3)

    def cos_output_shape(input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)

    x = Lambda(cos, output_shape=cos_output_shape)([x,y])
    return x

input_a = np.ones((4, 2, 3, 1))
input_b = np.ones((4, 2, 3, 1))

a = Input(batch_shape=(4, 2, 3, 1))
b = Input(batch_shape=(4, 2, 3, 1))

concat = merge([a, b], mode='concat', concat_axis=-1)
dot = merge([a, b], mode='dot', dot_axes=3)
cos = cos_sim(a,b)

model_concat = Model(input=[a, b], output=concat)
model_dot = Model(input=[a, b], output=dot)
model_cos = Model(input=[a, b], output=cos)

print(model_concat.predict([input_a, input_b]))
print(model_dot.predict([input_a, input_b]))
print(model_cos.predict([input_a, input_b]))