import numpy as np
import math
import matplotlib.pyplot as plt

def plot(data, title = None, x_label=None, y_label = None):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def generateGaussianDF(sigma, mu, show=False):    
    MAX = 256
    
    pdf = np.zeros(MAX)
    c = 1 / (math.sqrt( 2 * math.pi) * sigma)
    
    for x in range(MAX):
        x_minus_mu = x - mu
        
        power = (x_minus_mu ** 2) / (2 * sigma**2 )

        pdf[x] = c * math.exp(-power)# * 10000

    #pdf = pdf / np.min(pdf)

    if show:
        print("Generated Gaussian values")
        print(pdf)
    return pdf

def generateGaussianHistogram(sigma1, mu1, sigma2, mu2):
    df1 = generateGaussianDF(sigma=sigma1, mu=mu1)
    
    df2 = generateGaussianDF(sigma=sigma2, mu=mu2)
    
    #print(type(df2))
    #print(df2[0])
    
    res = df1.copy()
    
    for x in range(256):
        res[x] = df1[x] + df2[x]
        # if x % 10 == 0:
        #     print(f"{x}: {df1[x]} + {df2[x]} = {res[x]}")

    res = res / np.min(res) / 10
    return df1, df2, res

# df1, df2, res = generateGaussianHistogram(8,30,20,165)
# plot(df1)
# plot(df2)
# plot(res)
