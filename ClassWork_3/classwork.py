import cv2
import numpy as np
import math
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

def calculate_hist_pdf_cdf(image, input_hist = None):
    if input_hist is None:
        hist = cv2.calcHist([image], [0], None, [256],[0,256])
        dum = []
        for i in range(256):
            dum.append(hist[i][0])
        hist = dum
    else:
        hist = input_hist

    total = np.sum(hist)

    pdf = hist.copy()
    for i in range(len(hist)):
        pdf[i] = hist[i] / float(total)
    
    cdf = pdf.copy()
    for i in range(1,len(pdf)):
        cdf[i] = pdf[i] + cdf[i-1]
    
    return hist, pdf, cdf

def generate_mapping(input_cdf, target_cdf):
    in_map = np.zeros(256)
    
    for x in range(256):
        min_abs = math.inf
        inten = x     
        val = input_cdf[x]
        
        for y in range(256):
            dif = abs(val - target_cdf[y])
            if dif < min_abs:
                min_abs = dif
                inten = y
            # idx = np.abs(cdf - input_cdf[x]).argmin()
        in_map[x] = inten
    return in_map

def start(image):
    
    sigma1, mu1 = take_sigma_mu()
    sigma2, mu2 = take_sigma_mu(first=False)
    
    df1, df2, hist = generateGaussianHistogram(sigma1=sigma1,mu1=mu1,sigma2=sigma2,mu2=mu2)
    
    plot(data=df1,title='Gaussian - 1')
    plot(data=df2,title='Gaussian - 2')
    plot(data=hist,title='Double Gaussian Histogram')

    target_hist, target_pdf, target_cdf = calculate_hist_pdf_cdf(image=None, input_hist=hist)
    input_hist, input_pdf, input_cdf = calculate_hist_pdf_cdf(image=image)
    
    in_map = generate_mapping(input_cdf=input_cdf, target_cdf=target_cdf)

    h,w = image.shape
    output_image = image.copy()
    for x in range(h):
        for y in range(w):
            px_val = image[x,y]
            output_image[x,y] = in_map[ px_val ]

      
    cv2.imshow("Input Image", image)
    cv2.imshow("Output Image", output_image)
    # cv2.imwrite('.\images\\input_image.png',image)
    # cv2.imwrite('.\images\\output_image.png',output_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    out_hist, out_pdf, out_cdf = calculate_hist_pdf_cdf(image=output_image)
    
    plot(input_hist,title="Input hist")
    plot(input_pdf,title="Input pdf")
    plot(input_cdf,title="Input cdf")
    
    plot(out_hist,title="Output hist")
    plot(out_pdf,title="Output pdf")
    plot(out_cdf,title="Output cdf")
    

image = cv2.imread( "./images\\histogram.jpg", cv2.IMREAD_GRAYSCALE)
start(image=image)
