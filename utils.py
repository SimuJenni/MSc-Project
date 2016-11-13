import Queue as queue
import multiprocessing
import threading
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def imcrop_tosquare(img):
    """Cropping image to square keeping shorter dimension fixed.

    Args:
        img (np.ndarray): Image to be cropped

    Returns:
        crop: The cropped image of square size
    """
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop


def slice_montage(montage, img_h, img_w, n_imgs):
    """Slice a montage image into n_img h x w images.

    Performs the opposite of the montage function.  Takes a montage image and
    slices it back into a N x H x W x C image.

    Args:
        montage (np.ndarray): Montage image to slice.
        img_h (int): Height of sliced image
        img_w (int): Width of sliced image
        n_imgs (int): Number of images to slice

    Returns:
        sliced (np.ndarray): Sliced images as 4d array.
    """
    sliced_ds = []
    for i in range(int(np.sqrt(n_imgs))):
        for j in range(int(np.sqrt(n_imgs))):
            sliced_ds.append(montage[
                             1 + i + i * img_h:1 + i + (i + 1) * img_h,
                             1 + j + j * img_w:1 + j + (j + 1) * img_w])
    return np.array(sliced_ds)


def montage(images, saveto='montage.png', gray=False):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Args:
        images (np.ndarray): Images batch x height x width x channels
        saveto (str): Location to save the resulting montage image.

    Returns:
        m (np.ndarray): Montage image.
    """
    from PIL import Image

    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    if gray:
        m = Image.fromarray((m*255).astype(dtype=np.uint8))
        m.save(saveto)
    else:
        plt.imsave(arr=m, fname=saveto)
    return m


def im2float(imgs):
    return np.asarray(imgs, dtype='float32') / 255.


def generator_queue(generator, max_q_size=8, nb_worker=4, pickle_safe=False, wait_time=0.05):
    if pickle_safe:
        q = multiprocessing.Queue(maxsize=max_q_size)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()
    threads = []
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if pickle_safe or q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(nb_worker):
            if pickle_safe:
                # Reset random seed else all children processes share the same seed
                np.random.seed()
                thread = multiprocessing.Process(target=data_generator_task)
            else:
                thread = threading.Thread(target=data_generator_task)
            threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes
            for p in threads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop, threads