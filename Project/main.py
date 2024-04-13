import cv2
import numpy as np

import sys
sys.path.append("ClassWork_1")
sys.path.append("ClassWork_2")
sys.path.append("ClassWork_3")

from convolution import normalize
from canny import find_threeshold, perform_threshold, perform_hysteresis

from equalization import equalize
from helper import read_templates, perform_matching, show_image
from segmenter import segment_new, get_image_at_segment


def segment(image):
    thress = image.copy()
    image = image.copy()
    
    # show_image(image=image, name="atek")
    
    vert = []
    not_in_prev = True

    h,w = image.shape
        
    for x in range(h):
        if 0 not in image[x]: # contains no digit
            for y in range(w):
                image[x][y] = 100
            
            if not not_in_prev:
                vert.append(x)
            not_in_prev = True
        else:
            if not_in_prev:
                vert.append(x)
            not_in_prev = False
     
    not_in_prev = True
    horiz = []
    for y in range(w):
        if not np.any(image[:, y] == 0): # contains no digit
            for x in range(h):
                image[x][y] = 50
            if not not_in_prev:
                horiz.append(y)
            not_in_prev = True
        else:
            if not_in_prev:
                horiz.append(y)
            not_in_prev = False
    
    # print(horiz)
    # print(vert)
    
    rects = []
    n1 = len(horiz)
    n2 = len(vert)
    for i in range(0, n1, 2 ):
        for j in range(0, n2, 2 ):
            h1 = horiz[i]
            h2 = horiz[i+1]
            v1 = vert[j]
            v2 = vert[j+1]
            
            p1 = (h1, v1)
            p2 = (h2, v1)
            p3 = (h2, v2)
            p4 = (h1, v2)
            
            rect = (p1,p2,p3,p4)
            if is_rect_valid(thress,rect):
                rects.append(rect)
       
    return image, rects


def draw_point_at(image, x,y, color = 100):
    indices = ( (-1,-1), (0, -1), (1,-1), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1) )
    
    h,w = image.shape
    for xy in indices:
        nx = x + xy[0]
        ny = y + xy[1]
        if nx < 0 or ny < 0 or nx >= h or ny >= w:
            continue
        image[nx][ny] = color

def annotate_rect_points(image, my_segments):
    image = image.copy()
    
    for seg in my_segments:
        for pt in seg.rect:
            draw_point_at(image, pt[0], pt[1])
    return image

def extract_and_detect_segments(image, my_segments):
    for seg in my_segments:
        rect = seg.rect
        segment = get_image_at_segment(image, seg)
        matched_with = perform_matching(segment=segment)
        
        print("matched_with "+str(matched_with))
        show_image(segment)

def start_detection(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Gray scale conversion
    show_image(image=image, name="Gray scale converted")
    
    # Histogram equalize
    equalized_image = equalize(image, show = False)
    show_image(image=equalized_image, name="Histogram Equalized")
    
    threes = find_threeshold(image=equalized_image)
    # print(f"Threeshold: ${threes}")
    
    threes_image, weak, strong = perform_threshold(image=equalized_image,threes=threes)
    thress_image = perform_hysteresis( image=threes_image, weak=weak, strong=strong )
    
    thress_image = normalize(thress_image)
    show_image(thress_image, "Threesholded")
    
    # cropped, rects = segment(thress_image)
    cropped, my_segments = segment_new(thress_image)
    show_image(cropped, "Cropped")
    
    # for seg in my_segments:
    #     print(seg.rect)
    
    annotated = annotate_rect_points(cropped,my_segments)
    show_image(annotated,"Annotated")
    
    extract_and_detect_segments(thress_image, my_segments)


read_templates()
image = cv2.imread("Project\\images\\mis_aligned.png")
start_detection(image=image)
