import cv2
import numpy as np
import colorsys

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
from bg_remover import remove_background
from utility import crop_image, merge_images, get_actual_template
from aligner import get_aligned_digit

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

def draw_rect_at(image, top_left, top_right, bottom_right, bottom_left, INTEN):
    # top_left to top_right 
    
    for y in range(top_left[1], top_right[1]):
        image[ top_left[0], y ] = INTEN # -5

    # top_right to bottom_right
    for x in range(top_left[0], bottom_left[0]):
        image[ x, top_left[1]] = INTEN # +5

    # bottom_right to bottom_left
    for x in range(top_right[0], bottom_right[0]):
        image[ x, top_right[1]] = INTEN # x+5
    
    # bottom_left to top_left
    for y in range(bottom_left[1], bottom_right[1]):
        image[ bottom_left[0], y ] = INTEN #y-5

# points order is: lt, lb, rb, rt
def annotate_rect_points(image, my_segments):
    image = image.copy()
    
    for seg in my_segments:
        draw_rect_at(image, seg.rect[0], seg.rect[3], seg.rect[2], seg.rect[1],80)
    return image

def index_to_color(index):
    # Generate a unique hue by using the index. This keeps colors visually distinct.
    hue = (index * 0.618033988749895) % 1  # The golden ratio modulo 1 for good distribution
    # Convert hue, saturation, and value to RGB. Saturation and value are both set to 100%.
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # Convert from float (0 to 1 range) to an integer (0 to 255 range)
    return tuple(int(x * 255) for x in rgb)

def draw_text(image, text, top_left, bottom_right):
    text = str(text)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    text_x = top_left[1] + (bottom_right[1] - top_left[1] - text_width) // 2
    text_y = top_left[0] + (bottom_right[0] - top_left[0] + text_height) // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)


def extract_and_detect_segments(gray_image, my_segments):
    global gui
    
    color_image = np.stack((gray_image,)*3, axis=-1)

    height, width, _ = color_image.shape
    empty_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    index = 0
    for seg in my_segments:
        rect = seg.rect
        segment = get_image_at_segment(gray_image, seg)
        
        digit_one, digit_two = get_aligned_digit(segment.copy())
        
        matched_dig_act, matched_seg_act, percent_act, pcnts = perform_matching(segment=segment, use_actual=True)
        matched_dig_one, matched_seg_one, percent_one, _ = perform_matching(segment=digit_one)
        matched_dig_two, matched_seg_two, percent_two, _ = perform_matching(segment=digit_two)

        matched_dig = matched_dig_one
        matched_seg = matched_seg_one
        percent = percent_one
        
        if percent < percent_two:
            matched_dig = matched_dig_two
            matched_seg = matched_seg_two
            percent = percent_two
        
        if percent < percent_act:
            matched_dig = matched_dig_act
            matched_seg = matched_seg_act
            percent = percent_act
        
        if int(matched_dig) == 6 or int(matched_dig) == 9 and  abs(percent - percent_act) < 5: # 5% tolerable for 6,9
            matched_dig = matched_dig_act
            matched_seg = matched_seg_act
            percent = percent_act
        
        percent = "{:.3f}".format(percent)
        
        actual_template = get_actual_template(matched_dig)

        merged_image = merge_images(image1=segment, image2=digit_one, horiz=True)
        merged_image = merge_images(image1=merged_image, image2=digit_two, horiz=True)
        merged_image = merge_images(merged_image, actual_template, horiz = False)
        
        color = index_to_color(index)
        index += 1

        draw_rect_at(color_image, seg.rect[0], seg.rect[3], seg.rect[2], seg.rect[1], color) # BGR
        draw_rect_at(empty_image, seg.rect[0], seg.rect[3], seg.rect[2], seg.rect[1], color) # BGR
        
        draw_text(empty_image, matched_dig, seg.rect[0], seg.rect[2])
        
        gui.set_final_image(empty_image)
        
        # angle = calc_rotation_angle(re)
        
        # gui.add_frame(
        #     left_image=color_image, left_text='Matching with', 
        #     right_image=merged_image, right_text='Aligned image',
        #     bottom_text='Alinged image on right', keep_prev=True 
        # )
        
        gui.add_frame(
            left_image=color_image, left_text='Matching with',
            right_image=merged_image, right_text='Segment | Aligned-1 | Aligned-2 \n Best matched template',
            bottom_text=f"matched_with {str(matched_dig)} with percentage: {percent}."# Others are: 1:{pcnts[1]}, 2:{pcnts[2]}, 3:{pcnts[3]}, 4:{pcnts[4]}, 5:{pcnts[5]}, 6:{pcnts[6]}, 7:{pcnts[7]}, 8:{pcnts[8]}, 9:{pcnts[9]}, 0:{pcnts[0]}"
        )
        # show_image(segment)

norm_image = None
my_segments = None
def start_detection(image):
    global gui, norm_image, my_segments
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Gray scale conversion
    gui.add_frame(left_image=image, left_text='Gray scale converted')
    
    # image = remove_background(image)
    
    # Remove background
    image_without_back = remove_background(image)
    gui.add_frame(left_image=image_without_back, left_text='Background removed & Highlighted')
    
    # Crop image
    # image_cropped = crop_image(image_without_back)
    # gui.add_frame(left_image=image_cropped,left_text="Cropped image")    
    # threes = find_threeshold(image=image_without_back)
    # threes_image, weak, strong = perform_threshold(image=image_without_back,threes=threes)
    # thress_image = perform_hysteresis( image=threes_image, weak=weak, strong=strong )
    
    norm_image = normalize(image_without_back)
    # cv2.imwrite('D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\rotate_test\\rotated_44.png', thress_image)
    # gui.add_frame(left_image=thress_image, left_text='Threesholded', bottom_text=f"Threeshold value is: {threes}")

    cropped, my_segments = segment_new(image_without_back)
    gui.add_frame(left_image=cropped, left_text='Segmented image', bottom_text='Each isolated gray area is one segment')
    
    annotated = annotate_rect_points(cropped,my_segments)
    gui.add_frame(left_image=annotated, left_text='Annotated image') # Don't change this text idiot, used in gui part

def start_extract_and_detect_part():
    global norm_image, my_segments
    extract_and_detect_segments(norm_image, my_segments)
    
    global gui
    gui.show_message('Extracted. Continue to see')

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
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input\\no_back.png',
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input\\all_no_back.png',
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input\\odd_even.png',
            'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\Project\\images\\input\\all_solid_back.png'
        ],
        callback=Callbacks(start=on_start_req, on_ready= on_gui_ready, on_detect_start = start_extract_and_detect_part)
    )

start()










# Should work with solid background or having some background -done
# Try to align rotated version and then compare - done
# Try to use separate joing digit if possible - optional
# Show everything in GUI not in console - maybe color the current digit and show on right -done


# ------------------------------------ ---------------------- -----------------------------------
# Introduction
# Methodology - visual more - only core part
# Implementation - if any interface

# Final project report
# Last lab report + project report together
# Project demonstration(video or pic by pic) in slide

# Command for running:
# & D:\installation\Python\python.exe D:\Documents\COURSES\4.1\Labs\Image\ImageCodes\Project\main.py
