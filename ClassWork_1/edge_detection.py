import cv2
import numpy as np
import math

from convolution import convolve
from convolution import normalize
from kernal_generator import generateSobelKernel

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
    
    return (kernel, formatted_kernel)

def get_kernel():
    sigma = 0.7
    kernel,formatted_kernel = generateGaussianKernel(sigmaX=sigma, sigmaY=sigma, MUL=7)
    
    h = len(kernel)
    
    dr1 = np.zeros((h,h))
    dr2 = np.zeros((h,h))
    
    mn1 = 100
    mn2 = 100
    
    cx = h//2
    for x in range(h):
        for y in range(h):
            
            dx = (x-cx)
            dy = (y-cx)
            
            c1 = -dx / (sigma ** 2)
            c2 = -dy / (sigma ** 2)
            
            dr1[x,y] = c1 * kernel[x,y]
            dr2[x,y] = c2 * kernel[x,y]
            
            if dr1[x,y] != 0:
                mn1 = min(abs(dr1[x,y]),mn1)
            
            if dr2[x,y] != 0:
                mn2 = min(abs(dr2[x,y]),mn2)

    # dr1 = (dr1 / mn1).astype(int)
    # dr2 = (dr2 / mn2).astype(int)
    
    # print(dr1)
    # print(dr2)
    return (dr1,dr2)

def merge(image_horiz, image_vert):
    height, width = image.shape
    out = np.zeros_like(image, dtype='float32')
        
    for x in range(0, height):
        for y in range(0, width):
            dx = image_horiz[x,y]
            dy = image_vert[x,y]
                
            res = math.sqrt( dx**2 + dy**2 )
            out[x,y] = res

    out = normalize(out)
    return out

def find_avg(image, t = -1):
    total1 = 0
    total2 = 0
    c1 = 0
    c2 = 0
    
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x][y]
            if px > t:
                total2 += px
                c2 += 1
            else:
                total1 += px
                c1 += 1
    mu1 = total1 / c1
    mu2 = total2 / c2
    
    return (mu1 + mu2) / 2
            

def find_threeshold(image):
    total = 0
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x,y]
            total += px
    t = total / (h * w)
    
    dif = find_avg(image=image,t=t)
    while( abs(dif - t) < 0.1 ** 4 ) :
        t = dif
        dif = find_avg(image=image,t=t)
    
    return dif

def make_binary(t, image):
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x,y]
            
            image[x,y] = 255 if v > t else 0
    
    return image

image_path = '.\images\\lena.jpg'
image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)

kernel_x, kernel_y = get_kernel()

conv_1 = convolve(image=image, kernel=kernel_x)
conv_1_nor = normalize(conv_1)

conv_2 = convolve(image=image, kernel=kernel_y)
conv_2_nor = normalize(conv_2)

out = merge(conv_1, conv_2)
out_nor = normalize(out)

cv2.imshow("Vertical", conv_1_nor)
cv2.imshow("Horiz", conv_2_nor)
cv2.imshow("Merged", out_nor)

t = find_threeshold(image=out)
print(f"Threeshold {t}")
out = make_binary(t=t,image=image)
cv2.imshow("Threesholded", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
