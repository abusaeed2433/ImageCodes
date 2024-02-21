import cv2
import numpy as np

def normalize(image):
    cv2.normalize(image,image,0,255,cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def convolve(image, kernel, kernel_center = (-1,-1)):    
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    # if kernel center is not defined, then it will use the center symmetric center
    if kernel_center[0] == -1:
        kernel_center = ( kernel_width // 2, kernel_height // 2 )
    
    # pad the input image based on kernel and center
    padded_image = pad_image(image = image,  kernel_height = kernel_height, kernel_width = kernel_width, kernel_center = kernel_center)

    # generating output with dummy zeros(0)
    output = np.zeros_like(padded_image, dtype='float32')

    # padded image height, width
    padded_height, padded_width = padded_image.shape

    kcy = kernel_center[0]
    kcx = kernel_center[1]
    
    # iterating through height. For (1,1) kernel, it iterates from 1 to (h - 1)
    for y in range( kcy, padded_height - ( kernel_height - (kcy+1)) ):
        # iterate through width. For (1,1) kernel, it iterates from 1 to (w - 1)
        for x in range( kcx, padded_width - ( kernel_width - (kcx + 1)) ):
            
            # calculating the portion in image, that will be convoluted now
            image_start_x = x - kernel_center[1]
            image_start_y = y - kernel_center[0]
            
            sum = 0
            N = kernel_width // 2
            for kx in range( -N, N+1):
                for ky in range( -N, N+1 ):
                    rel_pos_in_kernel_x = kx + N # 0
                    rel_pos_in_kernel_y = ky + N # 0
                    
                    rel_pos_in_image_x = N - kx # 2
                    rel_pos_in_image_y = N - ky # 2
                    
                    act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
                    act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
                    
                    k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
                    i_val = padded_image[ act_pos_in_image_y ][ act_pos_in_image_x ]
                    
                    sum +=  k_val * i_val

                output[y,x] = sum

    # Crop the output to the original image size
    out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]
    
    return out

