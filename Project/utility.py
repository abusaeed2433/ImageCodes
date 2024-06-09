import cv2
import numpy as np


def get_actual_template(digit):
    path = "Project\images\\actual_template\\"+str(digit)+".png"
    image = cv2.imread(path,0)
    return image

def merge_images(image1, image2, horiz = False):

    height1, width1 = image1.shape
    height2, width2 = image2.shape

    new_height = 0
    new_width = 0

    if horiz:
        new_height = max(height1, height2)
        new_width = width1 + width2
    else:
        new_width = max(width1, width2)
        new_height = height1 + height2

    merged_image = 255 * np.ones( (new_height+4, new_width+4), dtype=np.uint8)

    # Copy over the first image
    for x in range(height1):
        for y in range(width1):
            merged_image[x][y] = image1[x][y]

    if horiz:
        for x in range(new_height):
            merged_image[x][width1] = 120
            merged_image[x][width1+1] = 120
    else:
        for y in range(new_width):
            merged_image[height1][y] = 120
            merged_image[height1+1][y] = 120

    # Copy over the second image\
    if horiz:
        for x in range(height2):
            for y in range(width2):
                merged_image[x][width1 + 2 +y] = image2[x][y]
    else:
        for x in range(height2):
            for y in range(width2):
                merged_image[height1 + 2 + x][y] = image2[x][y]

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
