import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

from convolution import normalize
from convolution import convolve
from kernal_generator import generateGaussianKernel

from edge_detection import get_kernel, merge, find_threeshold, make_binary

def perform_sobel(image):
    kernel_x, kernel_y = get_kernel()

    conv_x = convolve(image=image, kernel=kernel_x)
    conv_y = convolve(image=image, kernel=kernel_y)
    merged_image = merge(conv_x, conv_y)
    
    conv_x_nor = normalize(conv_x)
    conv_y_nor = normalize(conv_y)
    merged_image_nor = normalize(merged_image)
        
    t = find_threeshold(image=merged_image_nor)
    print(f"Threeshold {t}")
    final_out = make_binary(t=t,image=merged_image_nor, low = 0, high = 100)
    
    cv2.imshow("X derivative", conv_x_nor)
    cv2.imshow("Y derivative", conv_y_nor)
    
    cv2.imshow("Merged", merged_image_nor)
    cv2.imshow("Threesholded", final_out)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_out

def perform_canny(image_path, sigma):
    
    # Gray Scale Coversion
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    main_image = image.copy()
    
    # Perform Gaussian Blurr
    kernel, _ = generateGaussianKernel(sigmaX=sigma,sigmaY=sigma)
    image = convolve(image=image, kernel=kernel)
    image_nor = normalize(image=image)
    
    cv2.imshow("Blurred Input Image", image_nor)
    cv2.waitKey(0)
    
    # Gradient Calculation
    image = perform_sobel(image) 
    
    
    cv2.destroyAllWindows()
    

def start():
    #image_path = '.\images\\lena_again.png'
    image_path = '.\images\\medium.png'
    perform_canny(image_path=image_path, sigma=1)

start()
