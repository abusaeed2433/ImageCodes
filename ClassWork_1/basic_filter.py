import numpy as np
import cv2
import math
from kernal_generator import *
from extractor_merger import *
from convolution import normalize
from convolution import convolve
from enum import Enum
 
class InputImageType(Enum):
    GRAY = 1
    COLOR = 2
    HSV = 3
    
def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)
    
    return difference

def performConvol(imagePath, kernel, imageType = InputImageType.GRAY, kernel_center = (-1,-1)): # kernel_center = (y,x) # ( h, w )
    
    image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
    
    # cv2.imshow('input image in grayscale',image)
    # cv2.waitKey(0)
    
    out_conv = convolve(image=image, kernel=kernel, kernel_center=kernel_center)
    
    out_nor = normalize(out_conv)
    # print('After normalizing')
    # print(out)


    #out = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv2.imshow('Input image', image)
    cv2.imshow('Output image', out_nor)

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

#kernel = generateLogKernel(sigma = 1.4)
# print("LoG filter")
# print(kernel)

# kernel = generateSobelKernel()
# print("Sobel filter")
# print(kernel)

performConvol('.\images\\shape.jpg',kernel=kernel, imageType=InputImageType.GRAY, kernel_center = (1,1))


# image1 = cv2.imread('.\images\\table_1.jpg')
# image2 = cv2.imread('.\images\\table_2.jpg')
# dif = find_difference(image1=image1, image2=image2)

# cv2.imshow("Image_1", image1)
# cv2.imshow("Image_2", image2)
# cv2.imshow("Difference", dif)
# cv2.waitKey(0)

