import cv2
import numpy as np
from gaussian import generateGaussianHistogram, plot

def take_sigma_mu(first = True):
    print("Enter the value of sigma: ",end='')
    sigma = input()
    if sigma == '':
        sigma = 8 if first else 20
    else:
        sigma = float(sigma)
    
    print("Enter the value of mean: ",end='')
    mu = input()
    
    if mu == '':
        mu = 30 if first else 165
    else:
        mu = float(mu)
    
    return sigma, mu

def start(image):
    
    sigma1, mu1 = take_sigma_mu()
    sigma2, mu2 = take_sigma_mu(first=False)
    
    df1, df2, res = generateGaussianHistogram(sigma1=sigma1,mu1=mu1,sigma2=sigma2,mu2=mu2)
    
    plot(data=df1,title='Gaussian - 1', x_label='Value',y_label='Density')
    plot(data=df2,title='Gaussian - 2', x_label='Value',y_label='Density')
    plot(data=res,title='Double Gaussian Histogram', x_label='Value',y_label='Density')

image = cv2.imread( "./images\\histogram.jpg", cv2.IMREAD_GRAYSCALE)
start(image=image)
