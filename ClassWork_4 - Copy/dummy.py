import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def convert_to_complex(points):
    
    res = []
    for x,y in points:
        res.append( complex(x,y) )
    
    return res


import numpy as np

def apply_dft(complex_points):
    # Applying the DFT
    fourier_coefficients = np.fft.fft(complex_points)
    n = len(fourier_coefficients)

    frequencies = np.fft.fftfreq(n)

    amplitudes = np.abs(fourier_coefficients)
    phases = np.angle(fourier_coefficients)

    # indices = np.argsort(-amplitudes)

    return fourier_coefficients, frequencies

coordinates = [ (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10),
    (10,9), (10,8), (10,7),(10,6), (10,5), (10,4), (10,3), (10,2), (10,1), (9,1), (8,1), 
    (7,1), (6,1), (5,1), (4,1), (3,1), (2,1), (1,1)
]

def plot(points):
    x_values, y_values = zip(*points)

    # Create a scatter plot
    plt.scatter(x_values, y_values, color='b', label='Coordinates')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot of Coordinates')
    plt.grid(True)
    plt.legend()
    plt.show()

cmp_coordinates = convert_to_complex(coordinates)
print(cmp_coordinates)


# centering wrto to mean
x_list, y_list = zip(*coordinates)
x_mean = np.mean(x_list)
y_mean = np.mean(y_list)

tmp = []
for i in range(len(coordinates)):
    tmp.append( ( coordinates[i][0] - x_mean, coordinates[i][1] - y_mean) )

coordinates = tmp
# plot(coordinates)


# xlim_data = plt.xlim() 
# ylim_data = plt.ylim()

t_list = np.linspace(0, 360, len(x_list))
print(t_list)
print(len(t_list))


coeff, freq = apply_dft(cmp_coordinates)
# print(coeff)
# print(freq)

order = 1 
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order+1):
        new_coeffs.extend([coeffs[order+i],coeffs[order-i]])
    return np.array(new_coeffs)


sorted_coeff = sort_coeff(coeff)
# print(sorted_coeff)
# print( len(sorted_coeff) )

x_coeffs = np.real(coeff)
y_coeffs = np.imag(coeff)

draw_x, draw_y = [], []
def make_frame(i, time, coeffs):
    global pbar
    # get t from time
    t = time[i]

    # exponential term to be multiplied with coefficient 
    # this is responsible for making rotation of circle
    exp_term = np.array([np.exp(n*t*1j) for n in range(-order, order+1)])

    # sort the terms of fourier expression
    coeffs = sort_coeff(coeffs*exp_term) # coeffs*exp_term makes the circle rotate. 
    # coeffs itself gives only direction and size of circle

    # split into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    # center points for fisrt circle
    center_x, center_y = 0, 0

    # make all circles i.e epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # calculate radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff]) # similar to magnitude: sqrt(x^2+y^2)

        # draw circle with given radius at given center points of circle
        # circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        theta = np.linspace(0, tau, num=50) # theta should go from 0 to 2*PI to get all points of circle
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line, 

from matplotlib.animation import ImageMagickFileWriter


fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
anim = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50)
plt.show()
# anim.save('lines.mp4', writer=writer)
