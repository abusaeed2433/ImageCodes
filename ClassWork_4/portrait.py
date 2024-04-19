import cv2
import numpy as np
import matplotlib.pyplot as plt
from animator import start_animation

import sys
sys.path.append('.\ClassWork_2')

from canny import perform_canny


def get_edge_points(image):
    points = []
    
    h,w = image.shape
    
    for x in range(h):
        for y in range(w):
            if image[x,y] == 255:
                points.append((x,y))
    
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
    image = cv2.imread(image_path,0)
    
    edge = perform_canny(image=image, show=False)
    cv2.imshow("Cany result", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    cv2.imwrite("ClassWork_4\\images\\edge.png",edge)
    
    
    edge = cv2.imread("ClassWork_4\images\\edge.png",0)
    
    edge_points = get_edge_points(edge)
    print(len(edge_points))
    
    start_animation(edge_points)

image_path = "ClassWork_4\\images\\shape.jpg"
start(image_path)
