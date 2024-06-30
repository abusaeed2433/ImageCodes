import cv2
import numpy as np

TEMPLATE_SIZE = (60,80)

def show_image(image, name = "Image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def show_images(images, name = "Image"):
    i = 0
    for image in images:
        cv2.imshow(name+str(i), image)
        i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_templates(h=-1,w=-1, use_actual=False):
    templates = []
    for x in range(10):
        
        path = "Project\images\\min_area_template\\"+str(x)+".png"
        if use_actual:
            path = "Project\images\\actual_template\\"+str(x)+".png"

        image = cv2.imread(path,0)
        if h != -1 and w != -1:
            image = cv2.resize(image,(w,h),image)
        templates.append(image)
    return templates

def perform_matching(segment, use_actual = False):
    segment = segment.copy()
    templates = []
    
    h,w = segment.shape
    if h > TEMPLATE_SIZE[1] and w > TEMPLATE_SIZE[0]:
        segment = cv2.resize(segment,TEMPLATE_SIZE,segment)
        templates = read_templates(use_actual=use_actual)
        h = TEMPLATE_SIZE[1]
        w = TEMPLATE_SIZE[0]
        # print('Template not resized')
    else:
        # if h >= TEMPLATE_SIZE[1] or w >= TEMPLATE_SIZE[0]:
        h = min( TEMPLATE_SIZE[1], h)
        w = min( TEMPLATE_SIZE[0], w )
        
        # print('Template resized')
        segment = cv2.resize(segment,(w,h),segment)
        templates = read_templates(h=h, w=w, use_actual=use_actual)

    # print((h,w))
    # print(segment.shape)
    best_template = (-1,0)
    best_segment = None
    percents = []
    for ti in range( len(templates) ):
        template = templates[ti]
        # print("Shapes are")
        # print(template.shape)
        # print(segment.shape)
        
        matched = 0
        for x in range(h):
            for y in range(w):
                if template[x,y] == segment[x,y]:
                    matched += 1
        
        # percent = ( 100 * matched ) / (h * w)
        # print(percent)
        # show_images(images=[segment,template], name="Matched")
        # print(best_template)
        percents.append( (100 * matched) // (h*w) )
        if matched > best_template[1]:
            best_template = (ti, matched)
            best_segment = template
    
    percent = ( 100 * best_template[1] ) / (h * w)
    # print("percentage "+str(percent))
    return best_template[0], best_segment, percent, percents