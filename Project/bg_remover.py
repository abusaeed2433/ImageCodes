import cv2
import numpy as np
from collections import defaultdict

WHITE = 255
BLACK = 0
GRAY = 50

def remove_background(image):
    count_map = defaultdict(int)
    
    h,w = image.shape
    
    for x in range(h):
        for y in range(w):
            inten = image[x,y]
            count_map[inten] += 1
    
    mx = 0
    inten = 0
    for key in count_map.keys():
        if count_map[key] > mx:
            mx = count_map[key]
            inten = key
    
    new_image = image.copy()
    
    for x in range(h):
        for y in range(w):
            val = image[x,y]
            if val == inten:
                new_image[x,y] = WHITE
            # elif val == 2:
            #     new_image[x,y] = BLACK # making white to black to avoid conflict
            else:
                new_image[x,y] = GRAY
    return new_image
