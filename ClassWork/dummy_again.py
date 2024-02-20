import cv2
import numpy as np
from kernal_generator import *

import cv2
import numpy as np

def performConvol(imagePath, kernel, kernel_center):
    """
    Performs convolution on an image.

    Args:
    imagePath: The path of the image.
    kernel: The kernel to convolve with of type NumPy array.
    kernel_center: The centre position of the kernel as (y, x).

    Returns:
    The convoluted image as a NumPy array.
    """

    # Read the image
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # Extend image boundaries as needed
    image = extend_image_boundaries(image, kernel)

    # Get the dimensions of the kernel
    kernel_height, kernel_width = kernel.shape

    # Get the dimensions of the image
    image_height, image_width = image.shape

    # Initialize the output image
    output_image = np.zeros((image_height, image_width))

    # Iterate over every pixel in the image
    for y in range(image_height):
        for x in range(image_width):
            # Get the corresponding pixel in the kernel
            kernel_y = kernel_center[0] - y
            kernel_x = kernel_center[1] - x

            # Check if the kernel pixel is within the image boundaries
            if kernel_y >= 0 and kernel_y < kernel_height and kernel_x >= 0 and kernel_x < kernel_width:
                # Calculate the dot product of the kernel and the image patch
                dot_product = np.sum(kernel * image[y:y+kernel_height, x:x+kernel_width])

                # Add the dot product to the output image
                output_image[y, x] += dot_product

    # Normalize the result
    output_image = normalize_image(output_image)

    # Wrap the values into integer format
    output_image = output_image.astype(np.uint8)

    return output_image


def extend_image_boundaries(image, kernel):
    """
    Extends the boundaries of an image according to the kernel properties.

    Args:
    image: The image to extend.
    kernel: The kernel to convolve with.

    Returns:
    The extended image.
    """

    # Get the dimensions of the kernel
    kernel_height, kernel_width = kernel.shape

    # Calculate the amount of padding needed on each side
    pad_top = kernel_height // 2
    pad_bottom = kernel_height - pad_top - 1
    pad_left = kernel_width // 2
    pad_right = kernel_width - pad_left - 1

    # Extend the image boundaries
    image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)

    return image


def normalize_image(image):
    """
    Normalizes the pixel values of an image to be within 0 and 255.

    Args:
    image: The image to normalize.

    Returns:
    The normalized image.
    """

    # Get the minimum and maximum pixel values
    min_value = np.min(image)
    max_value = np.max(image)

    # Normalize the pixel values
    image = (image - min_value) / (max_value - min_value)

    # Return the normalized image
    return image


kernel_conv = (1/10) *np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

kernel_cor = (1/10)*np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

image = cv2.imread( 'ClassWork\\image_girl.jpg', cv2.IMREAD_GRAYSCALE )

out = cv2.filter2D(src=image, ddepth=-1, kernel = kernel_cor,borderType = cv2.BORDER_CONSTANT)
cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)

cv2.imshow("Actual", out)

my_out = performConvol('ClassWork\\image_girl.jpg', kernel = kernel_conv, kernel_center = (1,1))
cv2.imshow("Output", my_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
