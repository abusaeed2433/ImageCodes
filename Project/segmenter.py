import cv2
import numpy as np
from collections import deque

class MySegment:
    def __init__(self, rect, blackPoint):
        self.rect = rect
        self.blackPoint = blackPoint

def is_rect_valid(image,rect):
    
    top_left = rect[0]
    bottom_right = rect[2]
    
    PAD = 1
    for x in range( top_left[1]+PAD, bottom_right[1]-PAD ):
        for y in range( top_left[0]+PAD , bottom_right[0]-PAD ):
            if image[x,y] != 255:
                # print(image[x,y])
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
    
    black_points = []
    black_points.append( (x,y) )

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
            
        # indices = ( (-1,-1), (0, -1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1) ) # 8 connectivity
        indices = ( (0, -1), (-1,0), (0,0), (1,0), (0,1) ) # 4 connectivity
            
        for pt in indices:
            nx = x + pt[0]
            ny = y + pt[1]
            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue
            if image[nx,ny] == 0:
                queue.append( (nx,ny) )
                black_points.append( (nx,ny) )
    # print(area)
    # print(min_x, min_y, max_x, max_y)
    return min_x, min_y, max_x, max_y

def segment_new(image):
    image = image.copy()
    h,w = image.shape
    
    segments = []
    for x in range(h):
        for y in range(w):
            if image[x,y] == 0: # black
                # print(x,y)
                min_x, min_y, max_x, max_y = spread(image,x,y)
                
                p1 = (min_x, min_y)
                p2 = (max_x, min_y)
                p3 = (max_x, max_y)
                p4 = (min_x, max_y)
                
                rect = (p1,p2,p3,p4)
                segment = MySegment(rect=rect, blackPoint=(x,y))
                
                segments.append(segment)
    return image, segments

# returns only the segment excluding other area if any
def get_image_at_segment(image, seg):
    top_left = seg.rect[0]
    bottom_right = seg.rect[2]
    
    h = bottom_right[0] - top_left[0] + 2
    w = bottom_right[1] - top_left[1] + 2

    segment = np.ones((h,w))
    segment = segment * 255
    
    h,w = image.shape

    x = seg.blackPoint[0]
    y = seg.blackPoint[1]
    
    min_x = x
    max_x = x

    min_y = y
    max_y = y

    queue = deque()
    queue.append( (x,y) )
    
    sx = x - top_left[0]
    sy = y - top_left[1]
    segment[sx][sy] = 0
    
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
                sx = nx - top_left[0]
                sy = ny - top_left[1]
                try:
                    segment[sx][sy] = 0
                except Exception as e:
                    print(e)
    return segment