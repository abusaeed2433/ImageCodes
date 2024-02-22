import cv2
import numpy as np
from kernal_generator import generateSobelKernel

from convolution import *

import math
from extractor_merger import extract_rgb
from extractor_merger import extract_hsv
from extractor_merger import merge_rgb
from extractor_merger import merge_hsv
from extractor_merger import rgb_to_hsv

# gray = 1, rgb and hsv = 2
def perform_sobel(imagePath, conv_type = 1, kernel_center = (-1,-1)):
    if conv_type == 1:
        image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
        out = perform_two_sobel(image=image, kernel_center=kernel_center, show_output=True)
    else:
        image = cv2.imread(imagePath,cv2.IMREAD_COLOR)
        
        red, green, blue = extract_rgb(image)
        #blue, green, red = cv2.split(image)
        
        red_out = perform_two_sobel(image=red,kernel_center=kernel_center)
        green_out = perform_two_sobel(image=green,kernel_center=kernel_center)
        blue_out = perform_two_sobel(image=blue,kernel_center=kernel_center)
        
        red_out_nor = normalize(image=red_out)
        green_out_nor = normalize(image=green_out)
        blue_out_nor = normalize(image=blue_out)
        
        merged_rgb = merge_rgb(red=red_out,green=green_out, blue=blue_out)
        merged_rgb = normalize(merged_rgb)
        
        cv2.imshow("RED from RGB",red_out_nor)
        cv2.imshow("GREEN from RGB", green_out_nor)
        cv2.imshow("BLUE from RGB",blue_out_nor)
        cv2.imshow("Combined RGB",merged_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        image = rgb_to_hsv(image)
        hue, sat, val = extract_hsv(image=image)
        
        hue_out = perform_two_sobel(image=hue,kernel_center=kernel_center)
        sat_out = perform_two_sobel(image=sat,kernel_center=kernel_center)
        val_out = perform_two_sobel(image=val,kernel_center=kernel_center)
        
        hue_out_nor = normalize(hue_out)
        sat_out_nor = normalize(sat_out)
        val_out_nor = normalize(val_out)
        merged_hsv = merge_hsv(h=hue_out_nor, s=sat_out_nor, v=val_out_nor)
        
        cv2.imshow("HUE from HSV",hue_out_nor)
        cv2.imshow("SAT from HSV", sat_out_nor)
        cv2.imshow("VAl from HSV",val_out_nor)
        cv2.imshow("Combined HSV",merged_hsv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        merged_rgb_2 = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
        diff = find_difference(merged_rgb, merged_rgb_2)
        
        
        cv2.imshow("Original via RGB", merged_rgb)
        cv2.imshow("Merged via HSV", merged_rgb_2)
        cv2.imshow("Difference", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
def perform_two_sobel(image, kernel_center, show_output = False):
    kernel_horiz = generateSobelKernel(horiz=True)
    image_horiz = convolve(image=image, kernel=kernel_horiz, kernel_center=kernel_center)
    #image_horiz = normalize(image_horiz)
        
    if show_output:
        cv2.imshow('Horiz image', image_horiz)

    kernel_vert = generateSobelKernel(horiz=False)
    image_vert  = convolve(image=image, kernel=kernel_vert, kernel_center=kernel_center)
    #image_vert = normalize(image_vert)

    if show_output:
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
    
    if show_output:
        cv2.imshow('Output image', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out


def showSobelKernel():
    kernel_horiz = generateSobelKernel(horiz=True)
    print("Horizontal sobel kernel")
    print(kernel_horiz)

    kernel_vert = generateSobelKernel(horiz=False)
    print("Vertical sobel kernel")
    print(kernel_vert)

# image_path = '.\images\\lena.jpg'
# perform_sobel(imagePath=image_path, conv_type=2, kernel_center=(-1,-1))
