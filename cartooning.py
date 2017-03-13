from threading import Thread

import cv2
import numpy as np


def cartoonify(img_rgb, num_donw_samp=2, num_filter=100):
    """Cartoonify an image with bilateral filtering

    Args:
        img_rgb: Image as numpy array
        num_donw_samp: Number of down- and up-sampling steps (factor 2)
        num_filter: Number of times to apply bilateral filtering

    Returns:
        im: The cortoonified image

    """
    img_color = img_rgb
    im_shape = img_rgb.shape

    # Downsample
    for _ in range(num_donw_samp):
        img_color = cv2.pyrDown(img_color)

    # Repeatedly apply small bilateral filter
    for _ in range(num_filter):
        img_color = cv2.bilateralFilter(img_color, 9, sigmaColor=21, sigmaSpace=9)

    # Upsample
    for _ in range(num_donw_samp):
        img_color = cv2.pyrUp(img_color)

    img_color = cv2.resize(img_color, (im_shape[1], im_shape[0]))
    return img_color


def auto_canny(image, sigma=0.3, blur=3):
    im_shape = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.medianBlur(image, blur)

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    edged = cv2.resize(edged, (im_shape[1], im_shape[0]))

    # return the edged image
    return edged


def process_data(X, num_threads=10, num_downsample=2):
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
            X_toon[i, :, :] = cartoonify(X[i, :, :], num_donw_samp=num_downsample)
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
    import matplotlib.pyplot as plt

    img = cv2.imread("face.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (144, 144))

    cartoon = cartoonify(img_rgb, num_donw_samp=2)
    img_edge = auto_canny(img_rgb, sigma=0.99, blur=3)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_edge = img_edge.astype(dtype=np.uint8)
    print(img_edge.shape)
    print(cartoon.shape)
    print(img_rgb.shape)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5), sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})

    plt.gray()
    ax[0].imshow(img_rgb)
    ax[0].axis('off')
    ax[0].set_title('Original')
    ax[1].imshow(cartoon)
    ax[1].axis('off')
    ax[1].set_title('Cartoon')
    ax[2].imshow(img_edge)
    ax[2].axis('off')
    ax[2].set_title('Edge')
    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()

    plt.imsave('test_edge.jpg', img_edge)
    plt.imsave('test_toon.jpg', cartoon)
