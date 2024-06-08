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
from gui import create_gui, Callbacks, ImageGUI, Status


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

RECT_INTENSITY = 80

def draw_rect_at(image, top_left, top_right, bottom_right, bottom_left):
    # top_left to top_right 
    
    for y in range(top_left[1], top_right[1]):
        image[ top_left[0], y ] = RECT_INTENSITY # -5

    # top_right to bottom_right
    for x in range(top_left[0], bottom_left[0]):
        image[ x, top_left[1]] = RECT_INTENSITY # +5

    # bottom_right to bottom_left
    for x in range(top_right[0], bottom_right[0]):
        image[ x, top_right[1]] = RECT_INTENSITY # x+5
    
    # bottom_left to top_left
    for y in range(bottom_left[1], bottom_right[1]):
        image[ bottom_left[0], y ] = RECT_INTENSITY #y-5

# points order is: lt, lb, rb, rt
def annotate_rect_points(image, my_segments):
    image = image.copy()
    
    for seg in my_segments:
        draw_rect_at(image, seg.rect[0], seg.rect[3], seg.rect[2], seg.rect[1])
    return image

def extract_and_detect_segments(image, my_segments):
    return
    for seg in my_segments:
        rect = seg.rect
        segment = get_image_at_segment(image, seg)
        matched_with = perform_matching(segment=segment)
        
        print("matched_with "+str(matched_with))
        show_image(segment)

def start_detection(image):
    global gui
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Gray scale conversion
    # show_image(image=image, name="Gray scale converted")
    gui.add_frame(left_image=image, left_text='Gray scale converted')
    
    # Histogram equalize
    equalized_image = equalize(image, show = False)
    # show_image(image=equalized_image, name="Histogram Equalized")
    gui.add_frame(left_image=equalized_image, left_text='Histogram Equalized')
    
    threes = find_threeshold(image=equalized_image)
    
    threes_image, weak, strong = perform_threshold(image=equalized_image,threes=threes)
    thress_image = perform_hysteresis( image=threes_image, weak=weak, strong=strong )
    
    thress_image = normalize(thress_image)
    gui.add_frame(left_image=thress_image, left_text='Threesholded', bottom_text=f"Threeshold value is: {threes}")
    
    cropped, my_segments = segment_new(thress_image)
    gui.add_frame(left_image=cropped, left_text='Segmented image', bottom_text='Each isolated gray area is one segment')
    
    annotated = annotate_rect_points(cropped,my_segments)
    gui.add_frame(left_image=annotated, left_text='Annotated image')
    
    # extract_and_detect_segments(thress_image, my_segments)

def wait():
    global gui
    while not gui.wait_until():
        a = 10

def on_start_req(path):
    global gui
    gui.show_message(f'Image recevied by processor\n{path}')
    gui.show_button_text('Next Frame')
    gui.cur_state = Status.RUNNING

    image = cv2.imread(path)
    start_detection(image=image)


def on_gui_ready(l_gui):
    global gui
    gui = l_gui
    
    # gui.update_frame(left_image = )

gui:ImageGUI = None

def start():
    read_templates()
    
    create_gui(
        text_left='Input side',
        text_right='Output side',
        text_bottom='Message will shown here',
        image_paths=[
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input.png',
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\odd_even.png',
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input\\solid_back.png'
        ],
        callback=Callbacks(start=on_start_req, on_ready= on_gui_ready)
    )

start()

# Should work with solid background or having some background
# Try to align rotated version and then compare
# Try to use separate joing digit if possible - optional
# Show everything in GUI not in console - maybe color the current digit and show on right


# ------------------------------------ ---------------------- -----------------------------------
# Introduction
# Methodology - visual more - only core part
# Implementation - if any interface

# Final project report
# Last lab report + project report together
# Project demonstration(video or pic by pic) in slide
