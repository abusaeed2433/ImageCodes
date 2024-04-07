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

def read_templates(h=-1,w=-1):
    templates = []
    for x in range(10):
        path = "Project\images\\actual\\"+str(x)+".png"
        image = cv2.imread(path,0)
        if h != -1 and w != -1:
            image = cv2.resize(image,(w,h),image)
        templates.append(image)
    return templates

def perform_matching(segment):
    segment = segment.copy()
    templates = []
    
    h,w = segment.shape
    if h > TEMPLATE_SIZE[1] and w > TEMPLATE_SIZE[0]:
        segment = cv2.resize(segment,TEMPLATE_SIZE,segment)
        templates = read_templates()
    else:
        if h >= TEMPLATE_SIZE[1] or w >= TEMPLATE_SIZE[0]:
            segment = cv2.resize(segment,(w,h),segment)
        h = min( TEMPLATE_SIZE[1], h)
        w = min( TEMPLATE_SIZE[0], w )
        templates = read_templates(h=h, w=w)
        
    h = min( TEMPLATE_SIZE[1], h)
    w = min( TEMPLATE_SIZE[0], w )
    
    # print((h,w))
    # print(segment.shape)
    best_template = (-1,0)
    for ti in range( len(templates) ):
        template = templates[ti]
        matched = 0
        for x in range(h):
            for y in range(w):
                if template[x,y] == segment[x,y]:
                    matched += 1
        
        # percent = ( 100 * matched ) / (h * w)
        # print(percent)
        # show_images(images=[segment,template], name="Matched")
        # print(best_template)
        
        if matched > best_template[1]:
            best_template = (ti, matched)
    
    percent = ( 100 * best_template[1] ) / (h * w)
    print("percentage "+str(percent))
    return best_template[0]