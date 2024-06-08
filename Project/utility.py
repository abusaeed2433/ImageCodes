import cv2
import numpy as np

def get_median(neighborhood):
    sorted_neighborhood = sorted(neighborhood)
    median_index = len(sorted_neighborhood) // 2
    return sorted_neighborhood[median_index]

def get_neighborhood(x, y, img, filter_size):
    neighborhood = []
    for i in range(-(filter_size // 2), filter_size // 2 + 1):
        for j in range(-(filter_size // 2), filter_size // 2 + 1):
            if 0 <= x + i < img.shape[0] and 0 <= y + j < img.shape[1]:
                neighborhood.append(img[x + i, y + j])
    return neighborhood

def crop_image(image):
    height, width = image.shape
    
    top = 0
    while top < height:
        all_same = True
        inten = image[top][0]

        for y in range(width):
            if inten != image[top,y]:
                all_same = False
                break
        if not all_same:
            break
        top += 1

    bottom = height - 1
    while bottom >= 0:
        all_same = True
        inten = image[bottom][0]

        for y in range(width):
            if inten != image[bottom,y]:
                all_same = False
                break
        if not all_same:
            break
        bottom -= 1

    left = 0
    while left < width:
        all_same = True
        inten = image[0][left]

        for x in range(height):
            if inten != image[x,left]:
                all_same = False
                break
        if not all_same:
            break
        left += 1

    right = width -1
    while right >= 0:
        all_same = True
        inten = image[0][right]

        for x in range(height):
            if inten != image[x,right]:
                all_same = False
                break
        if not all_same:
            break
        right -= 1

    
    new_image = np.zeros( (bottom-top, right-left), dtype=np.uint8 )
    
    print(image.shape)
    print(new_image.shape)

    for x in range(top, bottom):
        for y in range(left, right):
            new_image[x-top, y-left] = image[x,y]
    
    return new_image
