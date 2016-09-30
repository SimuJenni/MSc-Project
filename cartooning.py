import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import cv2
from threading import Thread
import numpy as np


def cartoonify_TV1(im, numDownSamples=1, weight=0.1):

    # Downsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrDown(im)

    # Perform TVL1 denoising
    im = denoise_tv_chambolle(im, weight=weight, multichannel=True)

    # Upsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrUp(im)

    return im


def cartoonify_Bilateral(im, numDownSamples=1, numBilateralFilters=100):

    # Downsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrDown(im)

    # Repeatedly apply small bilateral filter
    for _ in xrange(numBilateralFilters):
        im = cv2.bilateralFilter(im, 8, 16, 7)

    # Upsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrUp(im)

    return im

def process_data(X, num_threads=10):
    X_proc = np.zeros_like(X)
    num_samples = X.shape[0]

    # Process data in num_threads equally sized slices
    slice_length = num_samples // num_threads
    starts = np.arange(0, num_samples-slice_length+1, slice_length)
    ends = np.arange(slice_length, num_samples+1, slice_length)

    # Store results of slices in a list
    results = [{} for x in range(0, num_threads)]
    def process_array(X, results, idx):
        num_im = X.shape[0]
        X_toon = np.zeros_like(X)
        for i in range(0, num_im):
            if i%1000==0:
                print('Thread {}: {}/{}'.format(idx, i, num_im))
            X_toon[i, :, :] = cartoonify_Bilateral(X[i, :, :])
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



def test_cartoon():
    im = plt.imread('knit.jpg')

    cartoon_BL = cartoonify_Bilateral(im)
    cartoon_TV = cartoonify_TV1(im)

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

if __name__ == '__main__':
    test_cartoon()

