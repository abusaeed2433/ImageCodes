import cv2
import numpy as np
from kernal_generator import *
from extractor_merger import extract_rgb
from extractor_merger import merge_rgb
from convolution import normalize
from convolution import convolve

def convolve_rgb(image, kernel, kernel_center=(-1,-1)):
    red, green, blue = extract_rgb(image)

    red_conv = convolve(image=red, kernel=kernel, kernel_center=kernel_center)
    red_nor = normalize(red_conv)
    
    green_conv = convolve(image=green, kernel=kernel, kernel_center=kernel_center)
    green_nor = normalize(green_conv)
    
    blue_conv = convolve(image=blue, kernel=kernel, kernel_center=kernel_center)
    blue_nor = normalize(blue_conv)
    
    cv2.imshow("Extracted Red", red_nor)
    cv2.imshow("Extracted green", green_nor)
    cv2.imshow("Extracted blue", blue_nor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    merged = merge_rgb(red=red_nor, green=green_nor, blue=blue_nor)
    cv2.imshow("Original image", image)
    cv2.imshow("Merged image", merged)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()


# image_path = '.\images\\shape.jpg'
# image = cv2.imread( image_path )

# kernel = np.array([
#     [0.00291502, 0.0130642,  0.02153923, 0.0130642,  0.00291502],
#     [0.0130642,  0.05854969, 0.09653213, 0.05854969, 0.0130642 ],
#     [0.02153923, 0.09653213, 0.15915457, 0.09653213, 0.02153923],
#     [0.0130642,  0.05854969, 0.09653213, 0.05854969, 0.0130642 ],
#     [0.00291502, 0.0130642,  0.02153923, 0.0130642,  0.00291502]
# ])

# convolve_rgb(image=image, kernel=kernel, kernel_center=(-1,-1))
