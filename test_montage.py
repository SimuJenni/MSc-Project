from DataGenerator import ImageDataGenerator
from ToonNet import Generator
from datasets import Imagenet
from utils import montage

batch_size = 32

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the auto-encoder
f_dims = [64, 128, 256, 512, 1024]
generator = Generator(data.dims, batch_size, load_weights=True, f_dims=f_dims, resize_conv=True, w_outter=True)

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size).next()
decoded_imgs = generator.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, 'Test-Out.jpeg')
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')