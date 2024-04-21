import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from animator import start_animation

import sys
sys.path.append('.\ClassWork_2')

from canny import perform_canny

def spread(image,x,y):
    h,w = image.shape
    
    queue = deque()
    queue.append( (x,y) )
    
    points = []
    points.append( (x,y) )

    while queue:
        item = queue.popleft()
        x = item[0]
        y = item[1]
            
        if image[x,y] != 255:
            continue

        image[x,y] = 120

        indices = ( (-1,-1), (0, -1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1) )

        for pt in indices:
            nx = x + pt[0]
            ny = y + pt[1]
            if nx < 0 or ny < 0 or nx >= h or ny >= w:
                continue
            if image[nx,ny] == 255:
                queue.append( (nx,ny) )
                points.append( (nx,ny) )

    return points

def get_contours(image):
    
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return contours


    contours = []
    
    image = image.copy()
    h,w = image.shape
    
    for x in range(h):
        for y in range(w):
            if image[x,y] == 255: # white
                points = spread(image,x,y)

                contours.append( points )
    
    return contours

    

def get_edge_points(image):
    contours = get_contours(image)
    
    W_F = 10
    H_F = 15
    
    points = []
    
    for cnt in contours:
        if len(points) >= len(cnt):
            continue
        points.clear()
        for pt in cnt:
            points.append( (pt[0][0]/W_F, pt[0][1]/H_F) )
            # points.append( (pt[0]/W_F, pt[1]/H_F) )
    return points


def convert_to_fourier(edge_coordinates):
    N = len(edge_coordinates)
    fourier_coefficients = []

    complex_signal = [x + 1j*y for x, y in edge_coordinates]

    for k in range(N):  # For each frequency component
        Xk = 0  # Initialize the k-th Fourier coefficient
        for n, x_n in enumerate(complex_signal):
            # Summation part of the DFT formula
            Xk += x_n * np.exp(-1j * 2 * np.pi * k * n / N)
        fourier_coefficients.append(Xk)

    return fourier_coefficients

def start(image_path):
    # image = cv2.imread(image_path,0)
    
    # edge = perform_canny(image=image, show=False)
    # cv2.imshow("Cany result", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    # cv2.imwrite("ClassWork_4\\images\\dog_edge.jpg",edge)
    
    
    image = cv2.imread("ClassWork_4\images\\dog.jpg",cv2.IMREAD_GRAYSCALE)
    
    edge_points = get_edge_points(image)
    
    print(f"no of edge points is: {len(edge_points)}")
    
    start_animation(edge_points)

image_path = "ClassWork_4\\images\\dog.jpg"
start(image_path)
