import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from animator import start_animation

import sys
sys.setrecursionlimit(10**3)
import threading
# threading.stack_size(2**26)
# threading.Thread(target=start).start()

sys.path.append('.\ClassWork_2')

from canny import perform_canny

def spread_2(image,sp_x,sp_y):
    h,w = image.shape
        
    parent_map = {}
    length = 0
    last = None
    
    def dfs(x,y, it):
        
        nonlocal length, last
        
        # if x == sp_x and y == sp_y and it != 0:
        #     return


        if image[x,y] != 255:
            image[x][y] = 120
            return
        
        image[x][y] = 120

        it += 1

        indices = [ (-1,-1), (0, -1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1) ]

        


        for pt in indices:
            nx = x + pt[0]
            ny = y + pt[1]
            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue

            if image[nx,ny] == 255:
                parent_map[(nx,ny)] = (x,y)
                dfs(nx,ny, it)
                if( length <= it):
                    length = it
                    last = (nx,ny)
            
                    # image[nx,ny] = 120

    parent_map[(sp_x,sp_y)] = None
    dfs(sp_x, sp_y, 0)

    print(length)

    points = []
    while( last != None ):
        points.append( last )
        last = parent_map[last]

    return points

def spread(image, sp_x, sp_y):
    h, w = image.shape
    
    parent_map = {}
    length = 0
    last = None
    
    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None
    
    while stack:
        x, y, it = stack.pop()

        # Skip this point if it's already processed or not white
        if image[x, y] != 255:
            continue
        
        # Mark this point as processed by setting it to 120
        image[x, y] = 120

        # Increment the path length
        it += 1

        # Store this as the last point if it's the longest path found so far
        if it > length:
            length = it
            last = (x, y)

        # Adjacent cell offsets
        temp = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        indices = []
        
        for pt in temp:
            pt2 = (pt[0]*2, pt[1]*2)
            
            indices.append(pt)
            indices.append(pt2)
        
        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and image[nx, ny] == 255:
                if (nx, ny) not in parent_map:  # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))
    
    # Retrieve the path points in order from last to first
    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]

    # Reverse to get points from first to last
    points.reverse()

    return points

def get_contours(image):
    
    # ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # return contours


    contours = []
    
    image = image.copy()
    
    height, width = image.shape[:2]
    margin = 10
    x1, y1 = margin, margin
    x2, y2 = width - margin, height - margin
    image = image[y1:y2, x1:x2]
    
    h,w = image.shape
    
    for x in range(h):
        for y in range(w):
            if image[x,y] == 255: # white
                points = spread(image,x,y)

                contours.append( points )
 
    return contours

def get_edge_points(image):
    contours = get_contours(image)
    print(f"Total contours found: {len(contours)}")
    
    W_F = 2
    H_F = 2
    
    image.fill(0)
    
    points = []

    for i in range(len(contours)):
        cnt = contours[i]
        
        if len(points) >= len(cnt):
            continue
        points.clear()
        
        for pt in cnt:
            points.append( (pt[0]/W_F, pt[1]/H_F) )

    # for i in range(len(points)):
    #     pt = points[i]
    #     image[pt[0]][pt[1]] = 255
    return points

def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def start(image_path):
    image = cv2.imread(image_path,0)
    
    edge = perform_canny(image=image, show=False)
    
    # edge = cv2.Canny(image, 50, 150)

    show_image("Cany result", image=edge)
    
    # kernel = np.ones((1, 2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    
    # kernel = np.ones((3,3),np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    
    # show_image("Eroded",image)
    
    # image = cv2.bitwise_not(image)
    
    # show_image("Inverted erosion", image)
    
    ## cv2.imwrite("ClassWork_4\\images\\dog_edge_final.jpg",edge)

    # edge = cv2.imread("ClassWork_4\images\\dog_edge_final.jpg", cv2.IMREAD_GRAYSCALE)
    
    show_image("Edge",image=edge)
    
    edge_points = get_edge_points(edge)
    
    print(f"no of edge points is: {len(edge_points)}")
    
    # exit(1)
    
    start_animation(edge_points)

image_path = "ClassWork_4\\images\\dog.jpg"
start(image_path)
