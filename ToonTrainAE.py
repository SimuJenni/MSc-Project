import os
import time
import gc
import sys
from ToonDataGenerator import ImageDataGenerator
from ToonNet import AE
from constants import MODEL_DIR, IMG_DIR
from datasets import CIFAR10_Toon, TinyImagenetToon
from utils import montage, generator_queue


# Get the data-set object
data = TinyImagenetToon()
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.75, 1.0],
    fill_mode='nearest',
    horizontal_flip=True
)

# Training parameters
batch_size = 200
nb_epoch = 10
chunk_size = 4 * batch_size
num_chunks = data.num_train // chunk_size
num_layers = 4
num_res = 0

ae, encoder, decoder = AE(data.dims, num_layers=num_layers, batch_size=batch_size)

# Name used for saving of model and outputs
print('Training network: {}'.format(encoder.name))

for epoch in range(nb_epoch):
    print('Epoch: {}/{}'.format(epoch, nb_epoch))

    # Create queue for training data
    data_gen_queue, _stop, threads = generator_queue(
        datagen.flow_from_directory(data.train_dir, batch_size=chunk_size, target_size=data.target_size),
        max_q_size=32,
        nb_worker=8)

    for chunk in range(num_chunks):

        # Get next chunk of training data from queue
        while not _stop.is_set():
            if not data_gen_queue.empty():
                toon_train, edge_train, img_train = data_gen_queue.get()
                break
            else:
                time.sleep(0.05)

        target = toon_train
        print('Epoch {}/{} Chunk {}: Training Discriminator...'.format(epoch, nb_epoch, chunk))

        # Train ae
        h = ae.fit(x=img_train, y=img_train, batch_size=batch_size, verbose=0, nb_epoch=1)
        print(h.history)
        sys.stdout.flush()

    _stop.set()
    del data_gen_queue, threads
    gc.collect()

# Save the model
encoder.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(encoder.name)))
decoder.save_weights(os.path.join(MODEL_DIR, '{}.hdf5'.format(decoder.name)))


# Generate montage of sample-images
sample_size = 100
X_test, Y_test = datagen.flow_from_directory(data.train_dir, batch_size=batch_size,
                                             target_size=data.target_size).next()
decoded_imgs = ae.predict(X_test, batch_size=batch_size)
montage(decoded_imgs[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Out.jpeg'.format(encoder.name)))
montage(X_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-X.jpeg'.format(encoder.name)))
montage(Y_test[:sample_size, :, :] * 0.5 + 0.5, os.path.join(IMG_DIR, '{}-Y.jpeg'.format(encoder.name)))
