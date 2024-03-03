import cv2
import numpy as np
from scipy import ndimage

from convolution import normalize, convolve

def perform_threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = image.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)
    
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def perform_hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == weak):
                if (
                    (image[i+1, j-1] == strong) or (image[i+1, j] == strong) or
                    (image[i+1, j+1] == strong) or (image[i, j-1] == strong) or
                    (image[i, j+1] == strong) or (image[i-1, j-1] == strong) or
                    (image[i-1, j] == strong) or (image[i-1, j+1] == strong)
                ):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


def start():
    image_path = 'images\suppressed.jpg'
    #image_path = 'images\lena.jpg'
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    
    res,weak,strong = perform_threshold(image)

    cv2.imshow("Threesholded image", normalize(res) )
    cv2.waitKey(0)

    final_output = perform_hysteresis( image=res, weak=weak, strong=strong )
    
    #final_output = final_output.astype(np.uint8)
    cv2.imshow("Final Edge", normalize(final_output))
    
    image_path = '.\images\\shape.jpg'
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    image_canny = cv2.Canny(image,50,50)
    cv2.imshow("Final Edge Library",image_canny)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(weak,strong)

#start()
