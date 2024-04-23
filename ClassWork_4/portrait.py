import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import sys
sys.path.append('.\ClassWork_2')

from animator import start_animation
from canny import perform_canny

def spread(image, sp_x, sp_y):
    h, w = image.shape
    
    parent_map = {}
    length = 0
    last = None
    
    stack = [(sp_x, sp_y, 0)]
    parent_map[(sp_x, sp_y)] = None
    
    while stack:
        x, y, it = stack.pop()

        # skip if not white
        if image[x, y] != 255:
            continue

        image[x, y] = 120

        it += 1

        if it > length:
            length = it
            last = (x, y)


        temp = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        indices = []
        
        for pt in temp:
            pt2 = (pt[0]*2, pt[1]*2)
            
            indices.append(pt)
            # indices.append(pt2)

        for dx, dy in indices:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and image[nx, ny] == 255:
                if (nx, ny) not in parent_map: # Check if not already visited
                    parent_map[(nx, ny)] = (x, y)
                    stack.append((nx, ny, it))

    # Get the longest path
    points = []
    while last is not None:
        points.append(last)
        last = parent_map[last]

    points.reverse()

    return points

def get_contours(image):
    image = image.copy()
    h,w = image.shape

    pad = 10
    image = image[pad:h-pad, pad:w-pad]
    h,w = image.shape

    contours = []
    
    to_it = (0,0)
    it = 0
        
    while to_it != None:
        
        stack = [to_it]

        to_it = None
        it += 1

        while stack:
            x, y = stack.pop()
            
            indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

            for dx, dy in indices:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= h or ny < 0 or ny >= w or image[nx, ny] == 120: # 120 = visited already
                    continue

                if image[nx, ny] == 255:
                    to_it = (nx, ny)
                    stack.clear()
                    break
                else:
                    image[nx, ny] = 120
                    stack.append( (nx,ny) )

        print(f"Going to iterate: {to_it}")
        if to_it != None:
            points = spread(image, to_it[0], to_it[1])
            if( len(points) > 50):
                contours.append( points )

            to_it = points[ len(points) - 1 ]
        if it == 1:
            break

    print(f"Total points iterated: {it}")
    
    show_image("After contour", image, wait=False)

    return contours

def get_edge_points(image):
    contours = get_contours(image)
    contours = sorted(contours, key=len)
    contours.reverse()
    
    print(f"Total contours found: {len(contours)}")

    W_F = H_F = 2

    points = []
    for i in range( min(3, len(contours) ) ):
        cnt = contours[i]
        # if len(points) >= len(cnt):
        #     continue
        # points.clear()

        for pt in cnt:
            points.append( (pt[0]/W_F, pt[1]/H_F) )
    return points

def show_image(name, image, wait=True):
    cv2.imshow(name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def start(image_path):
    image = cv2.imread(image_path,0)

    # edge = perform_canny(image=image, show=False)

    # show_image("Canny result", image=edge)

    # kernel = np.ones((1, 2), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    
    # kernel = np.ones((3,3),np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    
    # show_image("Eroded",image)
    
    # image = cv2.bitwise_not(image)
    
    # show_image("Inverted erosion", image)
    
    # cv2.imwrite("ClassWork_4\\images\\dog_edge_final.png",edge)

    edge = cv2.imread("ClassWork_4\images\\dog_edge_final.png", cv2.IMREAD_GRAYSCALE)

    edge_points = get_edge_points(edge)
    print(f"No of points for the maximum edge is: {len(edge_points)}")    

    start_animation(edge_points)

image_path = "ClassWork_4\\images\\dog.jpg"
start(image_path)
