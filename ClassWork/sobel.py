import cv2
import numpy as np
from kernal_generator import generateSobelKernel
from convolution import convolve
import math


def perform_sobel(imagePath, kernel_center = (-1,-1)):
    image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
    kernel_horiz = generateSobelKernel()
    image_horiz = convolve(image=image, kernel=kernel_horiz, kernel_center=kernel_center)
    
    #image_horiz = normalize(image_horiz)
    cv2.imshow('Horiz image', image_horiz)
    cv2.waitKey(0)

    kernel_vert = generateSobelKernel(horiz=False)
    image_vert  = convolve(image=image, kernel=kernel_vert, kernel_center=kernel_center)
    #image_vert = normalize(image_vert)
    
    cv2.imshow('Vertical image', image_vert)
    cv2.waitKey(0)
    
    height, width = image.shape
    out = np.zeros_like(image, dtype='float32')
    
    for x in range(0,height):
        for y in range(0, width):
            dx = image_horiz[x,y]
            dy = image_vert[x,y]
            
            res = math.sqrt( dx**2 + dy**2 )
            out[x,y] = res

    out = normalize(out)

    cv2.imshow('Output image', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

image_path = 'ClassWork/image_girl.jpg'
perform_sobel(imagePath=image_path, kernel_center=(0,0))