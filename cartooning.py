from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_tv_chambolle


def cartoonify_tv1(im, num_donw_samp=2, weight=0.1):
    """Cartoonify an image with total variation de-noising

    Args:
        im: Image as numpy array
        num_donw_samp: Number of down- and up-sampling steps (factor 2)
        weight: Parameter for total variation de-noising

    Returns:
        im: The cortoonified image

    """
    # Downsample
    for _ in range(num_donw_samp):
        im = cv2.pyrDown(im)

    # Perform TVL1 denoising
    im = denoise_tv_chambolle(im, weight=weight, multichannel=True)

    # Upsample
    for _ in range(num_donw_samp):
        im = cv2.pyrUp(im)

    return im


def cartoonify_bilateral(im, num_donw_samp=2, num_filter=100):
    """Cartoonify an image with bilateral filtering

    Args:
        im: Image as numpy array
        numDownSamples: Number of down- and up-sampling steps (factor 2)
        num_filter: Number of times to apply bilateral filtering

    Returns:
        im: The cortoonified image

    """
    # Downsample
    for _ in xrange(num_donw_samp):
        im = cv2.pyrDown(im)

    # Repeatedly apply small bilateral filter
    for _ in xrange(num_filter):
        im = cv2.bilateralFilter(im, 8, 16, 7)

    # Upsample
    for _ in xrange(num_donw_samp):
        im = cv2.pyrUp(im)

    return im


def process_data(X, num_threads=10):
    """Process an array of images with specified number of threads

    Args:
        X: numpy array (num_images, height, width, channels)
        num_threads: Number of threads

    Returns:
        X: with all images cartoonified
    """
    X_proc = np.zeros_like(X)
    num_samples = X.shape[0]

    # Process data in num_threads equally sized slices
    slice_length = num_samples // num_threads
    starts = np.arange(0, num_samples - slice_length + 1, slice_length)
    ends = np.arange(slice_length, num_samples + 1, slice_length)

    # Store results of slices in a list
    results = [{} for x in range(0, num_threads)]

    def process_array(X, results, idx):
        num_im = X.shape[0]
        X_toon = np.zeros_like(X)
        for i in range(0, num_im):
            if i % 1000 == 0:
                print('Thread {}: {}/{}'.format(idx, i, num_im))
            X_toon[i, :, :] = cartoonify_bilateral(X[i, :, :])
        results[idx] = X_toon

    # Create threads and feed slices
    threads = []
    for ii in range(0, num_threads):
        process = Thread(target=process_array, args=[X[starts[ii]:ends[ii], :, :], results, ii])
        process.start()
        threads.append(process)
    print('Processing all the images with {} threads'.format(num_threads))
    for process in threads:
        process.join()
    print('Done!')

    # Merge results
    for ii, value in enumerate(results):
        X_proc[starts[ii]:ends[ii], :, :] = value

    return X_proc


if __name__ == '__main__':
    im = plt.imread('landscape.jpg')

    cartoon_BL = cartoonify_bilateral(im)
    cartoon_TV = cartoonify_tv1(im)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5), sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})
    plt.gray()
    ax[0].imshow(im)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(cartoon_BL)
    ax[1].axis('off')
    ax[1].set_title('Bilateral')
    ax[2].imshow(cartoon_TV)
    ax[2].axis('off')
    ax[2].set_title('TV L1')
    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()
