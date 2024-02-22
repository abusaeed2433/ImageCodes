
import numpy as np
import cv2
from kernal_generator import *
from convolution import *

def test(img):
    blue_channel, green_channel, red_channel = cv2.split(img)
    
    horizontal_kernel = generateSobelKernel(horiz=True)
    vertical_kernel = generateSobelKernel(horiz=False)
    center = (-1,-1)
    
    red_horizontal_convolution = convolve(red_channel, horizontal_kernel, center)
    red_vertical_convolution = convolve(red_channel, vertical_kernel, center)
    
    output_red = sobelColvolution(red_channel,red_channel.shape,red_horizontal_convolution,red_vertical_convolution)
    cv2.normalize(output_red, output_red, 0, 255, cv2.NORM_MINMAX)
    output_red = np.round(output_red).astype(np.uint8)
    cv2.imshow("Red convoluted image",output_red)
    cv2.waitKey(0)
    

    # Convolution of green channel
    green_horizontal_convolution = grayscale_convolution.convolution(green_channel, horizontal_kernel, center)
    # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
    # cv2.waitKey(0)
    green_vertical_convolution = grayscale_convolution.convolution(green_channel, vertical_kernel, center)
    # cv2.imshow("Vertical convolution", red_vertical_convolution)
    output_green = sobelColvolution(green_channel, green_channel.shape, green_horizontal_convolution, green_vertical_convolution)
    cv2.normalize(output_green, output_green, 0, 255, cv2.NORM_MINMAX)
    output_green = np.round(output_green).astype(np.uint8)
    cv2.imshow("Green convoluted image",output_green)

    # Convolution of blue channel
    blue_horizontal_convolution = grayscale_convolution.convolution(blue_channel, horizontal_kernel, center)
    # cv2.imshow("Horizontal convolution", red_horizontal_convolution)
    blue_vertical_convolution = grayscale_convolution.convolution(blue_channel, vertical_kernel, center)
    # cv2.imshow("Vertical convolution", red_vertical_convolution)
    output_blue = sobelColvolution(blue_channel, blue_channel.shape, blue_horizontal_convolution, blue_vertical_convolution)
    cv2.normalize(output_blue, output_blue, 0, 255, cv2.NORM_MINMAX)
    output_blue = np.round(output_blue).astype(np.uint8)
    cv2.imshow("Blue convoluted image",output_blue)

    cv2.waitKey(0)

    output_RGB_image = cv2.merge([output_blue, output_green, output_red])
    cv2.imshow("RGB Convoluted image", output_RGB_image)
    cv2.waitKey(0)
    hsv_to_rgb_image = HSV_convolution.HSV_convolution(kernel_with_type, center)
    difference = cv2.absdiff(output_RGB_image, hsv_to_rgb_image)
    cv2.waitKey(0)
    cv2.imshow("Difference", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobelColvolution(img,size,hor,ver):
    height, width = size
    output = np.zeros_like(img, dtype='float32')

    for x in range(0, height):
        for y in range(0, width):
            dx = hor[x, y]
            dy = ver[x, y]

            result = math.sqrt(dx ** 2 + dy ** 2)
            output[x, y] = result
    return output

image_path = '.\images\\lena.jpg'
image = cv2.imread(image_path)
test(image)
