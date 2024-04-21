import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.integrate import quad_vec
import matplotlib.animation as animation

value_n = 200 # no of cycles = (2*value_n + 1)
circles = []
circle_lines = []
draw_x, draw_y = [], []
drawing = []
orig_drawing = []
x_list, y_list = [],[]

def start_animation(coordinates):
    global value_n, circles, circle_lines, drawing, orig_drawing, x_list, y_list
    
    y_list, x_list = zip(*coordinates)

    x_list = x_list - np.mean(x_list)
    y_list = y_list - np.mean(y_list)
    
    x_list, y_list = y_list, -x_list

    # visualize the contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_list, y_list)
    # plt.scatter(x_list, y_list, label='Data Points')

    # # later we will need these data to fix the size of figure
    xlim_data = plt.xlim()
    ylim_data = plt.ylim()
    
    plt.show()
        
    # plt.show()

    # time data from 0 to 2*PI as x,y is the function of time.
    t_list = np.linspace(0, 2*pi, len(x_list)) # now we can relate f(t) -> x,y

    print(t_list)

    
    
    # function to generate x+iy at given time t
    def f(t, t_list, x_list, y_list):
        return np.interp(t, t_list, x_list + 1j*y_list)

    print("generating coefficients ...")
    coeff = []
    for n in range(-value_n, value_n+1):
        coef = (1 / 2*pi) * quad_vec(
                lambda t: f(t, t_list, x_list, y_list)*np.exp(-n*t*1j), 
                0, 2*pi, # integral limit
                limit=100, full_output=1
            )[0] # first element is the integral value
        coeff.append(coef)

    coeff = np.array(coeff)
    print(coeff)


    # this is to store the points of last circle of epicycle which draws the required figure
    

    # make figure for animation
    fig, ax = plt.subplots()

    # different plots to make epicycle
    # there are -value_n to value_n numbers of circles
    circles = [ax.plot([], [], 'r-')[0] for i in range(-value_n, value_n+1)]

    # circle_lines are radius of each circles
    circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-value_n, value_n+1)]

    # drawing is plot of final drawing
    drawing, = ax.plot([], [], 'k-', linewidth=2)

    # original drawing
    orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

    # to fix the size of figure so that the figure does not get cropped/trimmed
    LIMIT = 500
    ax.set_xlim(xlim_data[0]-LIMIT, xlim_data[1]+LIMIT)
    ax.set_ylim(ylim_data[0]-LIMIT, ylim_data[1]+LIMIT)

    ax.set_axis_off() # hide axes
    ax.set_aspect('equal') # to have symmetric axes


    print("compiling animation ...")
    # set number of frames
    no_of_frames = 300
    
    time = np.linspace(0, 2*pi, num=no_of_frames)
    anim = animation.FuncAnimation(fig, make_frame, frames=no_of_frames, fargs=(time, coeff), interval=5)
    plt.show()


# save the coefficients in value_n 0, 1, -1, 2, -2, ...
# it is necessary to make epicycles
def sort_coeff(coeffs):
    global value_n
    
    new_coeffs = []
    new_coeffs.append(coeffs[value_n])
    for i in range(1, value_n+1):
        new_coeffs.extend([coeffs[value_n+i],coeffs[value_n-i]])
    return np.array(new_coeffs)

# make frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    global value_n, circles, circle_lines, draw_x, draw_y, drawing, orig_drawing, x_list, y_list
    
    t = time[i]

    # exponential term for all the circles
    exp_term = np.array([np.exp(n*t*1j) for n in range(-value_n, value_n+1)])

    # sort the terms of fourier expression. coeffs itself gives only direction and size of circle
    coeffs = sort_coeff(coeffs*exp_term) # coeffs * exp_term makes the circle rotate. 
    
    # split into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    center_x, center_y = 0, 0

    # make all circles
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        
        rad = (x_coeff**2 + y_coeff**2)**0.5

        # draw circle with given radius at given center points of circle
        # circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        
        theta = np.linspace(0, 2*pi, num=50) # theta should go from 0 to 2*PI to get all points of circle
        
        x, y = center_x + rad * np.cos(theta), center_y + rad * np.sin(theta)
        circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff
    
    # center points now are points from last circle
    # these points are used as drawing points
    draw_x.append(center_x)
    draw_y.append(center_y)

    # draw the curve from last point
    drawing.set_data(draw_x, draw_y)

    # draw the real curve
    orig_drawing.set_data(x_list, y_list)


coordinates = [ (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10),
    (10,9), (10,8), (10,7),(10,6), (10,5), (10,4), (10,3), (10,2), (10,1), (9,1), (8,1), 
    (7,1), (6,1), (5,1), (4,1), (3,1), (2,1), (1,1)
]

# start_animation(coordinates=coordinates)
