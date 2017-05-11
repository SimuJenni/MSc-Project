import cv2
import numpy as np


def cartooning(img_rgb, num_donw_samp=2, num_filter=100):
    """Cartoons an image with bilateral filtering

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

    return edged
