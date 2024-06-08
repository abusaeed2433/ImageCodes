import numpy as np
import cv2
import math
from math import inf

BLACK = 0
WHITE = 255


def fill_whole(old_image):
    h,w = old_image.shape
    image = old_image.copy()
    
    for x in range(h):
        for y in range(w):
            if old_image[x,y] != WHITE:
                image[x,y] = BLACK
                continue
            
            indices = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
            
            for pt in indices:
                nx = x + pt[0]
                ny = y + pt[1]
                if nx < 0 or ny < 0 or nx >= h or ny >= w:
                    continue
                if old_image[nx,ny] == BLACK:
                    image[x,y] = BLACK
                    break
    return image

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
        rotated_x = int(math.ceil(tx * cos_theta - ty * sin_theta))
        rotated_y = int(math.ceil(tx * sin_theta + ty * cos_theta))
        
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

def plot_points_on_image(h,w, best_points, left_x, top_y):
    image = 255 * np.ones((h+1,w+1), dtype=np.uint8)

    for pt in best_points:
        x = pt[0] - left_x
        y = pt[1] - top_y

        image[x][y] = BLACK
    return image

class RotatedState:
    def __init__(self,width, height, points, left_x, top_y, area, angle):
        self.width = width
        self.height = height
        self.points = points
        self.left_x = left_x
        self.top_y = top_y
        self.area = area
        self.angle = angle
    

def get_aligned_digit(segment): #segment is copied. Do whatever you want to
    points = extract_points(segment)
    temp_points = points
    best_points = points
    best_points_two = points
    
    w,h, lx, ly = get_area_param(points)
    
    min_area = w * h
    best_angle = 0
    
    rotated_states = []
    
    for angle in range(0,360, 2):
        points = rotate_points(temp_points, angle)
        nw, nh, nlx, nly = get_area_param(points)
        
        area = nw * nh

        if abs(min_area-area) <= 500:# and nw < w:
            rotated_states.append( RotatedState( nw, nh, points, nlx, nly, area, angle ) )
            
            min_area = area
            best_angle = angle
            best_points = points
            w = nw
            h = nh
            lx = nlx
            ly = nly

    rotated_states = sorted(rotated_states, key=lambda state: (state.width, state.area))
    
    image_one = None
    image_two = None
    
    if len(rotated_states) > 0:
        state = rotated_states[0]
        image_one = plot_points_on_image( state.height, state.width, state.points, state.left_x, state.top_y)
        image_one = fill_whole(image_one)

    if len(rotated_states) > 1:
        state = rotated_states[1]
        image_two = plot_points_on_image( state.height, state.width, state.points, state.left_x, state.top_y)
        image_two = fill_whole(image_two)
    
    # cv2.imshow('One', image_one)
    # cv2.imshow('Two', image_two)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_one, image_two

# image = cv2.imread('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\rotate_test\\rotated_1.png',0)
# cv2.imshow('Input image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image_one, image_two = get_aligned_digit(image)
# cv2.imwrite('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\rotate_test\\rotated_1_one.png', image_one)
# cv2.imwrite('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\rotate_test\\rotated_1_two.png', image_two)
