import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.integrate import quad_vec
import matplotlib.animation as animation

value_n = 200 # no of cycles = (2*value_n + 1)
circles = []
circle_lines = []
drawings = []
orig_drawings = []
x_lists, y_lists = [], []

def start_animation(coordinates_list):
    global value_n, circles, circle_lines, drawings, orig_drawings, x_lists, y_lists

    # Clear any previous data
    circles.clear()
    circle_lines.clear()
    drawings.clear()
    orig_drawings.clear()
    x_lists.clear()
    y_lists.clear()

    for coordinates in coordinates_list:
        y_list, x_list = zip(*coordinates)

        x_list = x_list - np.mean(x_list)
        y_list = y_list - np.mean(y_list)

        x_list, y_list = x_list, -y_list

        x_lists.append(x_list)
        y_lists.append(y_list)

    # visualize the contours
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x_list, y_list in zip(x_lists, y_lists):
        ax.plot(x_list, y_list)

    plt.show()

    # time data from 0 to 2*PI as x, y is the function of time
    t_list = np.linspace(0, 2*pi, len(x_lists[0]))

    # make figure for animation
    fig, ax = plt.subplots()

    # iterate over each list of points
    for x_list, y_list in zip(x_lists, y_lists):
        # function to generate x+iy at given time t
        def f(t, t_list, x_list, y_list):
            return np.interp(t, t_list, x_list + 1j*y_list)

        print("Generating coefficients ...")
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

        # there are -value_n to value_n numbers of circles
        circles.extend([ax.plot([], [], 'r-')[0] for _ in range(-value_n, value_n+1)])

        # circle_lines are radius of each circles
        circle_lines.extend([ax.plot([], [], 'b-')[0] for _ in range(-value_n, value_n+1)])

        # drawing is plot of final drawing
        drawings.append(ax.plot([], [], 'k-', linewidth=2)[0])

        # original drawing
        orig_drawings.append(ax.plot(x_list, y_list, 'g-', linewidth=0.5)[0])

    # to fix the size of figure so that the figure does not get cropped/trimmed
    LIMIT = 1000
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)

    ax.set_axis_off() # hide axes
    ax.set_aspect('equal') # to have symmetric axes

    print("compiling animation ...")
    # set number of frames
    no_of_frames = 300
    
    time = np.linspace(0, 2*pi, num=no_of_frames)
    anim = animation.FuncAnimation(fig, make_frame, frames=no_of_frames, fargs=(time,), interval=5)
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
def make_frame(i, time):
    global value_n, circles, circle_lines, drawings, orig_drawings, x_lists, y_lists
    
    for j, (x_list, y_list) in enumerate(zip(x_lists, y_lists)):
        t = time[i]

        # exponential term for all the circles
        exp_term = np.array([np.exp(n*t*1j) for n in range(-value_n, value_n+1)])

        # sort the terms of fourier expression. coeffs itself gives only direction and size of circle
        coeffs = sort_coeff(coefficients[j]*exp_term) # coeffs * exp_term makes the circle rotate. 
        
        # split into x and y coefficients
        x_coeffs = np.real(coeffs)
        y_coeffs = np.imag(coeffs)

        center_x, center_y = 0, 0

        # make all circles
        for k, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
            
            rad = (x_coeff**2 + y_coeff**2)**0.5

            # draw circle with given radius at given center points of circle
            # circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
            
            theta = np.linspace(0, 2*pi, num=50) # theta should go from 0 to 2*PI to get all points of circle
            
            x, y = center_x + rad * np.cos(theta), center_y + rad * np.sin(theta)
            circles[j*len(x_coeffs)+k].set_data(x, y)

            # draw a line to indicate the direction of circle
            x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
            circle_lines[j*len(x_coeffs)+k].set_data(x, y)

            # calculate center for next circle
            center_x, center_y = center_x + x_coeff, center_y + y_coeff
        
        # center points now are points from last circle
        # these points are used as drawing points
        draw_x, draw_y = [], []
        draw_x.append(center_x)
        draw_y.append(center_y)

        # draw the curve from last point
        drawings[j].set_data(draw_x, draw_y)

    return drawings
