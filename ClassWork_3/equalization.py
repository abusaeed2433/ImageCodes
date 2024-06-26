import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot(data,figure):
    plt.figure(figure) # Window number
    plt.plot(data)
    plt.show()
    
def show_all(image, c, show=True):
    hist = cv2.calcHist([image], [0], None, [256],[0,256])
    
    dum = []
    for i in range(256):
        dum.append(hist[i][0])
    hist = dum
    
    #print(hist.shape)
    #return None,None,None,None
        
    h, w = image.shape
    
    total = np.sum(hist)
    
    if show:
        cv2.imshow("Image",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    #print(len(hist))
    pdf = hist.copy()
    for i in range(len(hist)):
        pdf[i] = hist[i] / float(total)

    if show:
        plot(data=hist,figure=c+1)
        plot(data=pdf,figure=c+2)
    
    cdf = pdf.copy()
    
    for i in range(1,len(pdf)):
        cdf[i] = pdf[i] + cdf[i-1]
    
    if show:
        plot(cdf,c+3)
    
    in_map = cdf.copy()
    for i in range(len(cdf)):
        in_map[i] = np.round(cdf[i] * 255)
    
    return hist, pdf, cdf, in_map

def equalize(image, show = True):
    
    hist, pdf, cdf, in_map = show_all(image=image,c = 1, show=show)
    
    h,w = image.shape
    
    out = image.copy()
    for x in range(h):
        for y in range(w):
            px_val = image[x,y]
            out[x,y] = in_map[ px_val ]
    
    if show:
        _,_,_,_ = show_all(out,c=5,show=show)
        cv2.destroyAllWindows()
        
        cv2.imshow("Output",out)
        img2 = cv2.equalizeHist(image)
        cv2.imshow("Output_library",img2)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    return out


# image = cv2.imread( "ClassWork_3\\images\\histogram.jpg", cv2.IMREAD_GRAYSCALE)
# equalize(image=image, show=False)
