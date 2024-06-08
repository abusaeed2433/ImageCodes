import cv2
import numpy as np

def merge_images(image1, image2):
    height1, width1 = image1.shape
    
    height2, width2 = image2.shape

    max_width = max(width1, width2)

    # Create a new image of the appropriate size
    merged_image = 255 * np.ones((height1 + height2+3, max_width), dtype=np.uint8)

    # Copy over the first image
    for i in range(height1):
        for j in range(width1):
            merged_image[i][j] = image1[i][j]

    for j in range(width2):
        merged_image[height1 + 2][j] = 120

    # Copy over the second image
    for i in range(height2):
        for j in range(width2):
            merged_image[height1 + i][j] = image2[i][j]

    return merged_image


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
