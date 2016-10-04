import gc
import multiprocessing
from datetime import datetime

import numpy as np
import tensorflow as tf

from ToonNet import ToonNet
from datasets.Imagenet import Imagenet
from datasets.TinyImagenet import TinyImagenet

# constant
NUM_GPU = 2
NUM_EPOCH = 1
MINI_BATCH = 24


def train_model(gpu_id, data_queue, model_queue, input_shape, num_epoch, num_batch=1):
    with tf.device('/cpu:{}'.format(gpu_id)):
        # define the model
        model, encoder = ToonNet(input_shape=input_shape, batch_size=MINI_BATCH, out_activation='sigmoid',
                                 num_res_layers=10)

        # compile the model
        model.compile(optimizer='adam', loss='mse')

        # train the model
        best_loss = np.inf
        best_save = '_'.join((gpu_id, datetime.now().strftime('%Y_%m_%d_%H_%M_%S.h5')))

        for epoch in range(num_epoch):
            for batch in range(num_batch):
                print gpu_id, '@batch', batch
                data = data_queue.get()
                loss = model.train_on_batch(data[0], data[1])
                # after a batch a data, synchronize the model
                model_weight = [layer.get_weights() for layer in model.layers]
                # we need to send NUM_GPU-1 copies out
                for i in range(1, NUM_GPU):
                    model_queue[gpu_id].put(model_weight)
                for k in model_queue:
                    if k == gpu_id:
                        continue
                    # obtain the model from other GPU
                    weight = model_queue[k].get()
                    # sum it
                    for l, w in enumerate(weight):
                        model_weight[l] = [w1 + w2 for w1, w2 in zip(model_weight[l], w)]
                # average it
                for l, w in enumerate(model_weight):
                    model.layers[l].set_weights([d / NUM_GPU for d in w])

        # after each epoch, try to save the current best model
        if best_loss > loss:
            model.save_weights(best_save, overwrite=True)
            best_loss = loss
        model_queue[gpu_id].close()


if __name__ == '__main__':

    # Get the data-set object
    data = TinyImagenet()

    first_gpu_id = 2

    gpu_list = ['gpu{}'.format(i) for i in range(first_gpu_id, NUM_GPU)]
    # for send the data
    data_queue = multiprocessing.Queue(20)
    # for synchronize the model, we create a queue for each model
    model_queue = {gpu_id: multiprocessing.Queue(2) for gpu_id in gpu_list}

    threads = [multiprocessing.Process(target=train_model(gpu_id, data_queue, model_queue)) for gpu_id in gpu_list]

    for thread in threads:
        thread.start()

    for epoch in range(NUM_EPOCH):
        print 'data@epoch', epoch
        for X_train, Y_train in data.generator_train(MINI_BATCH):
            num_data = X_train.shape[0]
            for start in range(0, num_data, MINI_BATCH):
                print 'data@batch', start / MINI_BATCH
                data_queue.put((X_train[start:(start + MINI_BATCH)], Y_train[start:(start + MINI_BATCH)]))
            gc.collect()

    data_queue.close()

    for thread in threads:
        thread.join()
