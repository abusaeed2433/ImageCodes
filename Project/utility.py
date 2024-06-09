import cv2
import numpy as np

GAP_WIDTH = 40
GAP_WIDTH_BY_4 = 19
GAP_COLOR = 120
WHITE = 255

def get_actual_template(digit):
    path = "Project\images\\actual_template\\"+str(digit)+".png"
    image = cv2.imread(path,0)
    return image

def merge_horiz(image1, image2):
    height1, width1 = image1.shape
    height2, width2 = image2.shape

    new_height = max(height1, height2)
    new_width = width1 + width2

    merged_image = WHITE * np.ones( (new_height, new_width+GAP_WIDTH), dtype=np.uint8)

    pad = (new_height - height1)//2
    
    # Copy over the first image
    for x in range(height1):
        for y in range(width1):
            merged_image[x+pad][y] = image1[x][y]

    # vertical gap of width 4px
    for x in range(new_height):
        for i in range(GAP_WIDTH_BY_4, GAP_WIDTH-GAP_WIDTH_BY_4):
            merged_image[x][width1+i] = GAP_COLOR

    # Copy over the second image
    pad = (new_height - height2)//2
    for x in range(height2):
        for y in range(width2):
            merged_image[x+pad][width1 + GAP_WIDTH +y] = image2[x][y]

    return merged_image

def merge_vert(image1, image2):
    height1, width1 = image1.shape
    height2, width2 = image2.shape

    new_width = max(width1, width2)
    new_height = height1 + height2

    merged_image = WHITE * np.ones( (new_height+GAP_WIDTH, new_width), dtype=np.uint8)

    pad = (new_width - width1)//2
    # Copy over the first image
    for x in range(height1):
        for y in range(width1):
            merged_image[x][y+pad] = image1[x][y]
    
    # horiz gap of width 4px
    for y in range(new_width):
        for i in range(GAP_WIDTH_BY_4, GAP_WIDTH-GAP_WIDTH_BY_4):
            merged_image[height1+i][y] = GAP_COLOR

    # Copy over the second image
    pad = ( new_width - width2)//2
    for x in range(height2):
        for y in range(width2):
            merged_image[height1 + GAP_WIDTH + x][y+pad] = image2[x][y]

    return merged_image


def merge_images(image1, image2, horiz = False):
    if horiz:
        return merge_horiz(image1,image2)
    else:
        return merge_vert(image1, image2)

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
