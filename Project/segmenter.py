import cv2
import numpy as np
from collections import deque


def is_rect_valid(image,rect):
    
    top_left = rect[0]
    bottom_right = rect[2]
    
    PAD = 1
    for x in range( top_left[1]+PAD, bottom_right[1]-PAD ):
        for y in range( top_left[0]+PAD , bottom_right[0]-PAD ):
            if image[x,y] != 255:
                print(image[x,y])
                return True
    return False

def spread(image,x,y):
    h,w = image.shape
    
    min_x = x
    max_x = x
        
    min_y = y
    max_y = y
        
    queue = deque()
    queue.append( (x,y) )
    
    area = []
    while queue:
        item = queue.popleft()
        x = item[0]
        y = item[1]
            
        if image[x,y] != 0:
            continue
                    
        min_x = min( min_x, x )
        max_x = max( max_x, x )
            
        min_y = min( min_y, y )
        max_y = max( max_y, y )
        
        area.append((x,y))    
        image[x,y] = 120
            
        indices = ( (-1,-1), (0, -1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1) )
            
        for pt in indices:
            nx = x + pt[0]
            ny = y + pt[1]
            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue
            if image[nx,ny] == 0:
                queue.append( (nx,ny) )
    # print(area)
    # print(min_x, min_y, max_x, max_y)
    return min_x, min_y, max_x, max_y

def segment_new(image):
    image = image.copy()
    h,w = image.shape
    
    rects = []
    for x in range(h):
        for y in range(w):
            if image[x,y] == 0: # black
                print(x,y)
                min_x, min_y, max_x, max_y = spread(image,x,y)
                
                p1 = (min_x, min_y)
                p2 = (max_x, min_y)
                p3 = (max_x, max_y)
                p4 = (min_x, max_y)
                
                rect = (p1,p2,p3,p4)
                rects.append(rect)
    return image, rects
