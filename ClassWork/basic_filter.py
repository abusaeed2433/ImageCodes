import numpy as np
import cv2
import math
from kernal_generator import *
from enum import Enum
 
class InputImageType(Enum):
    GRAY = 1
    COLOR = 2
    HSV = 3

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_image

def performConvol(imagePath, kernel, imageType = InputImageType.GRAY, kernel_center = (-1,-1)):
    image = cv2.imread(
        imagePath, 
        cv2.IMREAD_GRAYSCALE
    )
    
    cv2.imshow('input image in grayscale',image)
    cv2.waitKey(0)
    
    # Pad the image based on the kernel size and center
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    if kernel_center[0] == -1:
        kernel_center = ( kernel_width // 2, kernel_height // 2 )
    
    padded_image = pad_image(image, kernel_height, kernel_width, kernel_center)

    # Output image, initialized to zeros with the size adjusted for padding
    output = np.zeros_like(padded_image, dtype='float32')

    # Adjusted image dimensions after padding
    padded_height, padded_width = padded_image.shape

    # Convolution process
    for y in range(kernel_center[0], padded_height - kernel_height + kernel_center[0] + 1):
        for x in range(kernel_center[1], padded_width - kernel_width + kernel_center[1] + 1):
            sum = 0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    posY = y - kernel_center[0] + ky
                    posX = x - kernel_center[1] + kx
                    sum += kernel[ky][kx] * padded_image[posY][posX]
            output[y, x] = sum

    # Crop the output to the original image size
    out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]
            
    print('Before normalizing')
    print(out)
    
    cv2.waitKey(0)
    cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    print('After normalizing')
    print(out)

    #out = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imshow('Output image', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


kernel = generateGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5, cx = -1, cy = -1 )
# print("Gaussian filter")
# print(kernel)

# kernel = generateMeanKernel()
# print("Mean filter")
# print(kernel)

# kernel = generateLaplacianKernel()
# print("Laplacian filter")
# print(kernel)

# kernel = generateLogKernel(sigma = 1.4)
# print("LoG filter")
# print(kernel)

# kernel = generateSobelKernel()
# print("Sobel filter")
# print(kernel)

performConvol('box.jpg',kernel=kernel, imageType=InputImageType.GRAY, kernel_center = (0,0))
#performConvol('image_girl.jpg',kernel=kernel, imageType=InputImageType.GRAY)
