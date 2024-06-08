import tkinter as tk
from tkinter import Label, Frame, Tk, Button
from tkinter.ttk import Separator
import cv2
from PIL import Image, ImageTk
from tkinter.ttk import Separator, Style
from tkinter.font import Font
from tkinter import Label, Frame, Tk, Button, OptionMenu, StringVar
import os
from enum import Enum

class Callbacks:
    def __init__(self, start, on_ready, on_detect_start):
        self.start = start
        self.on_ready = on_ready
        self.on_detect_start = on_detect_start

class Status(Enum):
    NOT_STARTED = 1
    INPUT_IMAGE_SELECTED = 2
    RUNNING = 3
    ENDED = 4
    DONE_TILL_DETECTION = 5

class MyFrame:
    def __init__(self, left_image=None, right_image=None, left_text = None, right_text = None, bottom_text = None):
        self.left_image = left_image
        self.right_image = right_image
        
        if self.left_image is not None:
            self.left_image = left_image.copy()
        
        if self.right_image is not None:
            self.right_image = right_image.copy()
        
        self.left_text = left_text
        self.right_text = right_text
        self.bottom_text = bottom_text

class ImageGUI:
    def __init__(self, root, text_left, text_right, text_bottom, image_paths, callback, image_left = None, image_right =None):
        self.root = root
        self.frames = []
        self.cur_index = -1

        self.cur_state = Status.NOT_STARTED
        self.root.title("Digit Recognizer")
        self.root.state('zoomed')
        self.callback = callback
        
        self.text_to_show = 'Text will appear here'
    
        # Create custom fonts
        self.custom_font_large = Font(family="MS Sans Serif", size=14, weight="bold")
        self.custom_font_small = Font(family="MS Sans Serif", size=12)

        self.create_left_right_image_frame()
        self.create_separators()
        self.create_text_frame()
        self.create_button()
        self.set_drop_down(image_paths)

        # Set grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(1, weight=6)
        self.root.grid_rowconfigure(0, weight=1)

        
        # Init data
        self.add_frame(left_text=text_left, right_text=text_right, bottom_text=text_bottom)
        self.show_next_frame()
        
        self.callback.on_ready(self)

    def add_frame(self, left_image=None, right_image=None, left_text = None, right_text = None, bottom_text = None):
        frame = MyFrame(left_image=left_image, right_image=right_image, left_text=left_text, right_text=right_text, bottom_text=bottom_text)
        self.frames.append(frame)
    
    def on_prev_frame_show(self):
        if self.cur_index - 1 < 0:
            self.show_message('No previous frame found')
            return
        
        self.cur_state = Status.RUNNING
        self.cur_index -= 1
        
        frame = self.frames[self.cur_index]
        
        self.update_frame(
            left_image = frame.left_image,
            right_image = frame.right_image,
            left_text = frame.left_text,
            right_text = frame.right_text,
            bottom_text = frame.bottom_text
        )
        
        
    def show_next_frame(self):
        if self.cur_index + 1 >= len(self.frames):
            self.show_message('Please wait. Next frame is not ready')
            return False

        self.cur_index += 1
        
        frame = self.frames[self.cur_index]
        
        self.update_frame(
            left_image = frame.left_image,
            right_image = frame.right_image,
            left_text = frame.left_text,
            right_text = frame.right_text,
            bottom_text = frame.bottom_text
        )
        return True
    
    def update_frame(self, left_image = None, right_image = None, left_text = None, right_text = None, bottom_text = None):
        
        self.update_image(left_image, self.left_image_label, right_side=False)
        self.update_image(right_image, self.right_image_label, right_side=True)
        
        if left_text is None:
            left_text = ''

        self.text_display_left.config(text=left_text)
        
        if right_text is None :
            right_text = ''

        self.text_display_right.config(text=right_text)
            
        if bottom_text is None :
            bottom_text = ''

        self.text_display_bottom.config(text=bottom_text)
    def update_image(self, image, label, right_side):
        if image is None:
            label.configure(image='')
            label.image = None
        else:
            if not right_side:
                image = cv2.resize(image, (500, 300))
            
            img_pil = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            label.configure(image=img_tk)
            label.image = img_tk

    def create_left_right_image_frame(self):
        # LEFT
        self.left_frame = Frame(self.root, bd=2)
        self.left_frame.grid(row=1, column=0, sticky='nsew')
        self.left_frame.grid_propagate(False) 

        # RIGHT
        self.right_frame = Frame(self.root, bd=2)
        self.right_frame.grid(row=1, column=2, sticky='nsew')
        self.right_frame.grid_propagate(False) 
        
        # Labels for the images
        self.left_image_label = Label(self.left_frame)
        self.left_image_label.pack(fill=tk.BOTH, expand=True)
        self.right_image_label = Label(self.right_frame)
        self.right_image_label.pack(fill=tk.BOTH, expand=True)
        
        self.left_image_label.grid_propagate(False) 
        self.right_image_label.grid_propagate(False) 

    def create_separators(self):
        # VERTICAL
        sep_vert = Separator(self.root, orient='vertical', style='Dark.TSeparator')
        sep_vert.grid(row=0, column=1, rowspan=3, sticky='ns')
        
        for i in range(4,7,2):
            sep_vert = Separator(self.root, orient='vertical', style='Dark.TSeparator')
            sep_vert.grid(row=i, column=1, rowspan=1, sticky='ns')
        
        for i in range(4, 8, 2):
            sep_horiz = Separator(self.root, orient='horizontal', style='Dark.TSeparator')
            sep_horiz.grid(row=i, column=0, columnspan=3, sticky='ew')

    def create_text_frame(self):
        # TEXT LEFT
        self.text_frame_left = Frame(self.root, bd=2)
        self.text_frame_left.grid(row=0, column=0, columnspan=1, sticky='nsew')
        
        self.text_display_left = Label(self.text_frame_left, text='-',bg='blue', fg='white', font=self.custom_font_large)
        self.text_display_left.pack(fill=tk.BOTH, expand=True)
        
        # TEXT RIGHT
        self.text_frame_right = Frame(self.root, bd=2)
        self.text_frame_right.grid(row=0, column=2,sticky='nsew')
        
        self.text_display_right = Label(self.text_frame_right, text='-', bg='red', fg='white', font=self.custom_font_large)
        self.text_display_right.pack(fill=tk.BOTH, expand=True)
        
        # TEXT BOTTOM
        self.text_frame_bottom = Frame(self.root, bd=2)#, relief='solid')
        self.text_frame_bottom.grid(row=5, column=0, columnspan=3, sticky='nsew')
        
        self.text_display_bottom = Label(self.text_frame_bottom, text='-', bg='black', fg='white', font=self.custom_font_small)
        self.text_display_bottom.pack(fill=tk.BOTH, expand=True)
    
    def create_button(self):
        self.text_frame = Frame(self.root, bd=2)
        self.text_frame.grid(row=7, column=2, columnspan=2, sticky='nsew')
        
        self.text_frame.grid_configure(sticky='ew')  # Expand the frame horizontally
        
        self.btn_prev = Button(self.text_frame, text="Prev frame", command=self.on_prev_frame_show, bg='blue', fg='white')
        self.btn_prev.pack(side='left', expand=True, pady=8)
        
        self.btn_next = Button(self.text_frame, text="Next Frame", command=self.on_click, bg='blue', fg='white')
        self.btn_next.pack(side='left', expand=True, padx=5, pady=8)  # Also set side to 'left'
    
    def on_click(self):
        if (self.cur_state == Status.RUNNING):
            if not self.show_next_frame(): # failed to show frame
                current_text = self.text_display_left.cget('text')
                if(current_text == 'Annotated image'): # start annotation part
                    self.show_message('Please wait. Starting next part....')
                    self.callback.on_detect_start()
            return

        if self.cur_state == Status.NOT_STARTED:
            
            self.show_message('Select an image first')
            return
    
        if self.cur_state == Status.ENDED:
            
            self.show_message('Select image again')
            return
        
        if self.cur_state == Status.INPUT_IMAGE_SELECTED:
            self.show_message('Processing image....')
            self.callback.start(self.selected_image_path)
            return

        self.show_message('Processing....')
    
    def get_names_and_mapping(self, image_paths):
        name_path_mapping = {}
        names = []
        for path in image_paths:
            file_name = os.path.basename(path)
            names.append(file_name)
            
            name_path_mapping[file_name] = path
        return names, name_path_mapping

    def set_drop_down(self, image_paths):
        self.selected_option = StringVar(self.root)
        self.selected_option.set("Select input image")
        
        self.names, self.name_path_mapping = self.get_names_and_mapping(image_paths)
        
        self.option_menu = OptionMenu(self.root, self.selected_option, *self.names, command=self.option_selected)
        self.option_menu.grid(row=7, column=0, columnspan=2, sticky='ew', padx=10, pady=10)
        
        # print(self.names)
        # print(self.name_path_mapping)
    
    def show_message(self, message):
        #if message is not None :
        self.text_display_bottom.config(text=message)
        
    def show_button_text(self, text):
        self.btn_next.config(text=text)

    def option_selected(self, value):
        print(f"You selected {value}")
        
        self.selected_image_path = self.name_path_mapping[value]
        self.cur_state = Status.INPUT_IMAGE_SELECTED
        
        image = cv2.imread(self.selected_image_path)
        self.add_frame(left_image=image,left_text='Input image')
        self.show_next_frame()
        
        self.show_button_text("Start detecting")
        self.show_message(f"Selected: {value}")


def create_gui(text_left, text_right, text_bottom, image_paths, callback):
    root = Tk()
    # image = cv2.imread('Project/images/input/solid_back.png')
    app = ImageGUI( root, text_left, text_right, text_bottom, image_paths, callback)
    root.mainloop()

