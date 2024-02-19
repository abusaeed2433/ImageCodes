import numpy as np


img = np.array(
        [
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]
        ]
    )

img2 = np.array(
        [
        [1,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,0,1],
        [16,17,18,19,20],
        [21,22,23,24,25]
        ]
    )

kernel = np.array(
        [
        [1,2,3],
        [4,5,6],
        [7,8,9]
        ]
    )



output = np.zeros((5,5))

padded_height, padded_width = 5,5

kcy = 1
kcx = 1

def conv_at(x,y, kc = (0,0)):
    mx = -kc[0]
    my = -kc[1]
    
    image_start_x = x - kc[0]
    image_start_y = y - kc[1]
    image_end_x = image_start_x + 3
    image_end_y =  image_start_y + 3
    
    print(image_start_x, image_start_y)
    print(image_end_x, image_end_y)
    
    sum = 0
    N = 3//2
    for kx in range( -N, N+1):
        for ky in range( -N, N+1 ):
            rel_pos_in_kernel_x = kx + N # 0
            rel_pos_in_kernel_y = ky + N # 0
            
            
            rel_pos_in_image_x = N - kx # 2
            rel_pos_in_image_y = N - ky # 2
            
            act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
            act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
            
            k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
            i_val = img[ act_pos_in_image_x ][ act_pos_in_image_y ]
            
            sum +=  k_val * i_val
            
            print( "(", rel_pos_in_kernel_x, ",", rel_pos_in_kernel_y, ") * (", act_pos_in_image_x, ",", act_pos_in_image_y, "): ", k_val, "*", i_val  )
        output[x,y] = sum

# for y in range( kcy, padded_height - (3-(kcy+1)) ):
#     for x in range( kcx, padded_width - ( 3 - (kcx + 1)) ):


print( kernel[-1,-1] )

conv_at(2,3, (0,1))
