from DataGenerator import ImageDataGenerator, Y2X_Y2Y
from ToonNet import Encoder
from datasets import Imagenet
from utils import montage

batch_size = 32

# Get the data-set object
data = Imagenet()
datagen = ImageDataGenerator()

# Load the auto-encoder
encoder, generator = Encoder(data.dims, load_weights=True, w_outter=True)

# Generate montage of sample-images
sample_size = 64
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=sample_size, xy_fun=Y2X_Y2Y).next()
decoded_imgs = generator.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, 'Test-Out.jpeg')
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-X.jpeg')
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, 'Test-Y.jpeg')

print(encoder.layers[1])
print(generator.layers[1])