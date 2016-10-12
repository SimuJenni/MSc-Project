from DataGenerator import ImageDataGenerator
from ToonNet import ToonAE
from datasets.Imagenet import Imagenet
from utils import montage

batch_size = 32

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator(horizontal_flip=True)

# Load the auto-encoder
toonAE = ToonAE(input_shape=data.dims, batch_size=batch_size)
#toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAEGAN.hdf5')
toonAE.load_weights('/home/sj09l405/MSc-Project/ToonAE.hdf5')
toonAE.compile(optimizer='adam', loss='mae')

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size).next()
decoded_imgs = toonAE.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, 'Test-Out.jpeg')
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')