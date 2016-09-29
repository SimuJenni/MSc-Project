import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import cv2


def cartoonify_TV1(im, numDownSamples=2, weight=0.1):

    # Downsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrDown(im)

    # Perform TVL1 denoising
    im = denoise_tv_chambolle(im, weight=weight, multichannel=True)

    # Upsample
    for _ in xrange(numDownSamples):
        im = cv2.pyrUp(im)

    return im


def cartoonify_Bilateral(im, numDownSamples=2, numBilateralFilters=50):

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


def test_cartoon():
    im = plt.imread('pizza.jpg')
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

