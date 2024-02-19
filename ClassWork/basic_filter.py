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
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def performConvol(imagePath, kernel, imageType = InputImageType.GRAY, kernel_center = (-1,-1)): # kernel_center = (y,x) # ( h, w )
    
    image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
    
    cv2.imshow('input image in grayscale',image)
    cv2.waitKey(0)
    
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    if kernel_center[0] == -1:
        kernel_center = ( kernel_width // 2, kernel_height // 2 )
    
    padded_image = pad_image(image, kernel_height, kernel_width, kernel_center)

    output = np.zeros_like(padded_image, dtype='float32')

    padded_height, padded_width = padded_image.shape

    kcy = kernel_center[0]
    kcx = kernel_center[1]
    
    
    for y in range( kcy, padded_height - ( kernel_height - (kcy+1)) ):
        for x in range( kcx, padded_width - ( kernel_width - (kcx + 1)) ):
                
            image_start_x = x - kernel_center[0]
            image_start_y = y - kernel_center[1]
            
            sum = 0
            NH = kernel_height // 2
            NW = kernel_width // 2
            for kx in range( -NH, NH+1):
                for ky in range( -NW, NW+1 ):
                    rel_pos_in_kernel_x = kx + NH # 0
                    rel_pos_in_kernel_y = ky + NW # 0
                    
                    rel_pos_in_image_x = NH - kx # 2
                    rel_pos_in_image_y = NW - ky # 2
                    
                    act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
                    act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
                    
                    k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
                    i_val = image[ act_pos_in_image_x ][ act_pos_in_image_y ]
                    
                    sum +=  k_val * i_val

                output[x,y] = sum

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

performConvol('ClassWork\\box.jpg',kernel=kernel, imageType=InputImageType.GRAY, kernel_center = (0,0))
#performConvol('image_girl.jpg',kernel=kernel, imageType=InputImageType.GRAY)
