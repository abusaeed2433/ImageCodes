import numpy as np
import cv2
import math
from math import inf

BLACK = 0

def extract_points(segment):
    points = []
    
    h,w = segment.shape
    
    for x in range(h):
        for y in range(w):
            if(segment[x,y] == BLACK):
                points.append( (x,y) )
    return points

def rotate_points(points, angle):
    angle_rad = math.radians(angle)
    ox, oy = (0,0)
    rotated_points = []
    cos_theta, sin_theta = math.cos(angle_rad), math.sin(angle_rad)

    for x, y in points:
        tx, ty = x - ox, y - oy
        
        # Apply rotation
        rotated_x = tx * cos_theta - ty * sin_theta
        rotated_y = tx * sin_theta + ty * cos_theta
        
        # Reverse translation
        final_x = rotated_x + ox
        final_y = rotated_y + oy
        
        rotated_points.append((final_x, final_y))
    
    return rotated_points

# Returns the horizontal and vertical rectangle area w,h needed to cover all the points
def get_area_param(points):
    min_x = inf
    min_y = inf
    max_x = -inf
    max_y = -inf

    for pt in points:
        min_x = min(min_x, pt[0])
        min_y = min(min_y, pt[1])
        
        max_x = max(max_x, pt[0])
        max_y = max(max_y, pt[1])

    w = max_y - min_y
    h = max_x - min_x
    
    return w, h, min_x, min_y # Area

def get_aligned_digit(segment): #segment is copied. Do whatever you want to
    
    points = extract_points(segment)
    best_points = points
    
    w,h, lx, ly = get_area_param(points)
    
    min_area = w * h
    best_angle = 0
    
    print(min_area)
    
    for angle in range(0,180, 5):
        points = rotate_points(points, 5)
        nw, nh, nlx, nly = get_area_param(points)
        
        area = nw * nh
        
        # print(f'Best: {min_area}, Cur:{area}')
        if min_area > area:
            min_area = area
            best_angle = angle
            best_points = points
            w = nw
            h = nh
            lx = nlx
            ly = nly
    
    image = np.zeros((h+1,w+1), dtype=np.uint8)
    
    for pt in best_points:
        x = pt[0] - lx
        y = pt[1] - ly
        
        image[x][y] = BLACK
    
    print(best_angle)
    return image
    
image = cv2.imread('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\actual\\2.png',0)
cv2.imshow('Input image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

aligned = get_aligned_digit(image)
cv2.imshow('Rotated image', aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
