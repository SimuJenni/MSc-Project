import os
from constants import CIFAR10_DATADIR
from keras.datasets import cifar10
from cartooning import process_data
import h5py


def load_cartoon_data():
    # First check if dir exists and create if not
    if not os.path.exists(CIFAR10_DATADIR):
        os.makedirs(CIFAR10_DATADIR)

    file_name = 'dataset.hdf5'
    data_path = os.path.join(CIFAR10_DATADIR, file_name)

    # Cartooned data already exist -> load and return it
    if os.path.isfile(data_path):
        with h5py.File(data_path, 'r') as data:
            return (data['X_train'][:], data['Y_train'][:]), (data['X_test'][:], data['Y_test'][:])

    # Otherwise get the data set and process it
    (Y_train, _), (Y_test, _) = cifar10.load_data()
    X_train = process_data(Y_train)
    X_test = process_data(Y_test)

    # Save the processed data and return
    with h5py.File(data_path, 'w') as data:
        data.create_dataset('X_train', data=X_train)
        data.create_dataset('Y_train', data=Y_train)
        data.create_dataset('X_test', data=X_test)
        data.create_dataset('Y_test', data=Y_test)
        return (data['X_train'][:], data['Y_train'][:]), (data['X_test'][:], data['Y_test'][:])


if __name__ == '__main__':
    load_cartoon_data()