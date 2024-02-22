import cv2
import numpy as np

from convolution import convolve
from convolution import normalize

def generateGaussianKernel(sigmaX, sigmaY, MUL = 5):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1
    
    #print(w,h)

    cx = w // 2
    cy = h // 2 

    kernel = np.zeros((w, h))
    c = 1 / ( 2 * 3.1416 * sigmaX * sigmaY )
    
    for x in range(w):
        for y in range(h):
            dx = x - cx
            dy = y - cy
            
            x_part = (dx*dx) / (sigmaX * sigmaX)
            y_part = (dy*dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp( - 0.5 * (x_part + y_part) )

    formatted_kernel = kernel / np.min(kernel)
    formatted_kernel = formatted_kernel.astype(int)

    print("Formatted gaussian filter")
    print(formatted_kernel)
    
    return (kernel, formatted_kernel)

#def start(image):
    

kernel,formatted_kernel = generateGaussianKernel(sigmaX=0.7, sigmaY=1)
print(kernel)

image_path = '.\images\\lena.jpg'
image = cv2.imread(image_path,)
start()