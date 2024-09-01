from PIL import Image, ImageDraw, ImageFont
import cv2

def write_text_on_image(image_path, output_path, text):
    # Load the image
    image = cv2.imread(image_path)

    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)  # White color
    thickness = 2

    # Get the text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to place the text
    x = (image.shape[1] - text_width) // 2
    y = (image.shape[0] + text_height) // 2

    # Draw the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

    # Save the image with the text
    cv2.imwrite(output_path, image)

def start():
    root = 'C:\\Users\\User\\Downloads\\transit_hub\\'
    
    MAX = 1
    for i in range(MAX):
        for j in range(MAX):
            input = root + 'blank.png'
            output = root + f'zoom_4\\zoom_FOUR_({i}_{j}).png'
            
            write_text_on_image(input, output, f'({i*MAX + j+1})\n 4')

start()    
# write_text_on_image('C:\\Users\\User\\Downloads\\transit_hub\\blank.png', 'C:\\Users\\User\\Downloads\\transit_hub\\output.png', 'Hello')
