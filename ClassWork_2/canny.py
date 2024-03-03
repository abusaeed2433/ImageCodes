import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

from convolution import normalize
from convolution import convolve
from kernal_generator import generateGaussianKernel
from three_hys import perform_threshold, perform_hysteresis

from edge_detection import get_kernel, merge, find_threeshold, make_binary

def perform_sobel(image):
    kernel_x, kernel_y = get_kernel()

    conv_x = convolve(image=image, kernel=kernel_x)
    conv_y = convolve(image=image, kernel=kernel_y)
    
    kernel, _ = generateGaussianKernel(sigmaX=1,sigmaY=1,MUL=5)
    conv_x = convolve(image=conv_x,kernel=kernel)
    conv_y = convolve(image=conv_y,kernel=kernel)
    
    merged_image = merge(conv_x, conv_y)
    
    #theta = np.arctan2(conv_x, conv_y)
    theta = np.arctan2( conv_y.copy(), conv_x.copy() )
    
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
    
    #return final_out, theta
    return merged_image, theta

def perform_non_maximum_suppression(image, theta):
    image = image.copy()
    
    image = image / image.max() * 255
    
    M, N = image.shape
    #out = np.zeros((M,N), dtype=np.int32) # resultant image
    out = np.zeros((M,N), dtype=np.uint8)
    
    angle = theta * 180. / np.pi        # max -> 180, min -> -180
    #angle[angle < 0] += 180             # max -> 180, min -> 0

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 0
            r = 0
            
            ang = angle[i,j]
            
            if ( -22.5 <= ang < 22.5) or ( 157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j-1]
                q = image[i, j+1]

            elif (  -67.5 <= ang <= -22.5 ) or ( 112.5 <= ang <= 157.5):
                r = image[i-1, j+1]
                q = image[i+1, j-1]

            elif ( 67.5 <= ang <= 112.5) or ( -112.5 <= ang <= -67.5 ):
                r = image[i-1, j]
                q = image[i+1, j]

            elif ( 22.5 <= ang < 67.5 ) or ( -167.5 <= ang <= -112.5 ):
                r = image[i+1, j+1]
                q = image[i-1, j-1]

            if (image[i,j] >= q) and (image[i,j] >= r):
                out[i,j] = image[i,j]
            else:
                out[i,j] = 0
    return out

def perform_canny(image_path, sigma):
    
    # Gray Scale Coversion
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    main_image = image.copy()
    
    # Perform Gaussian Blurr
    kernel, _ = generateGaussianKernel(sigmaX=sigma,sigmaY=sigma,MUL=7)
    image = convolve(image=image, kernel=kernel)
    
    cv2.imshow("Blurred Input Image", normalize(image))
    cv2.waitKey(0)
    
    # Gradient Calculation
    image_sobel, theta = perform_sobel(image) 
    
    # Non Maximum Suppression
    suppressed = perform_non_maximum_suppression(image=image_sobel,theta=theta)
    
    # Threesholding and hysteresis
    threes, weak, strong = perform_threshold(image=suppressed)
    final_output = perform_hysteresis( image=threes, weak=weak, strong=strong )
        
    cv2.imshow("After sobel", normalize(image_sobel) )
    cv2.imshow("Non maximum suppression", normalize(suppressed) )
    cv2.imshow("Threesholded", normalize(threes) )
    cv2.imshow("Final", normalize(final_output))

    
    # cv2.imwrite('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\ClassWork_2\\images\\suppressed.jpg',suppressed)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    

def start():
    # image_path = '.\images\\lena_again.png'
    # image_path = '.\images\\medium.png'
    image_path = '.\images\\shape.jpg'
    # image_path = '.\images\\lena_git.jpg'
    perform_canny(image_path=image_path, sigma=1)

start()
