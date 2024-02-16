import numpy as np
import math
import numpy as np
import scipy.ndimage as ndimage

def generateGaussianKernel(sigmaX, sigmaY, MUL = 7, cx = -1, cy = -1):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1
    
    #print(w,h)

    if cx == -1:
        cx = w // 2
    if cy == -1:
        cy = h // 2

    kernel = np.zeros((w, h))
    c = 1 / ( 2 * 3.1416 * sigmaX * sigmaY )
    
    for x in range(w):
        for y in range(h):
            dx = x - cx
            dy = y - cy
            
            x_part = (dx*dx) / (sigmaX * sigmaX)
            y_part = (dy*dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp( - 0.5 * (x_part + y_part) )

    formatted_kernel = kernel / np.min(kernel)
    formatted_kernel = formatted_kernel.astype(int)

    #print("Formatted gaussian filter")
    #print(formatted_kernel)
    
    return kernel

def generateMeanKernel(rows = 3, cols = 3):
    kernel = np.zeros( (rows, cols) )

    for x in range(0, rows):
        for y in range(0, cols):
            kernel[x,y] = 1.0
            
    #print("Formatted mean kernel")
    #print(kernel)
    return kernel / (rows * cols)

def generateLaplacianKernel( negCenter = True ):
    n = 3    
    other_val = 1 if negCenter else -1
    
    kernel = other_val * np.ones( (n, n) )
    center = n // 2
    
    kernel[center, center] = - other_val * ( n*n - 1 )
    
    #print(kernel)
    return kernel

def generateLogKernel(sigma, MUL = 7):
    n = int(sigma * MUL)
    n = n | 1
    
    kernel = np.zeros( (n,n) )

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    
    mn = math.inf
    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            
            part2 = (dx**2 + dy**2) / (2 * sigma**2)
            
            kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)
            mn = min( abs(kernel[x][y]), mn )
    
    #print("Formatted LoG kernel")
    
    #print( (kernel / mn).astype(int) )
    
    return kernel

def generateSobelKernel( horiz = True ):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    sobel_y = np.array([
        [1, 2, 1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    
    return sobel_x if horiz else sobel_y

def testKernel():
    #kernel = generateGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5, cx = -1, cy = -1 )
    #print(kernel)
    
    #kernel = generateMeanKernel( rows = 3, cols = 3 )
    #print(kernel)
    
    #kernel = generateLaplacianKernel( negCenter = True )
    #print(kernel)
    
    #kernel = generateLogKernel(1.4)
    #print(kernel)
    
    #kernel = generateSobelKernel( horiz = False )
    #print(kernel)
    
    print("I don't know")


testKernel()
