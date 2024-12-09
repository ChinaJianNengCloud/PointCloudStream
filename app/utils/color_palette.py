import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_num_of_palette(num):
    """
    Returns a list of RGB colors of length 'num'.
    If 'num' is less than or equal to the length of ORIGINAL_GOLIATH_PALETTE,
    it returns the first 'num' colors from the palette.
    If 'num' is greater, it extends the palette by generating additional colors.
    """
    ORIGINAL_GOLIATH_PALETTE = [
        [50, 50, 50],
        [255, 218, 0],
        [102, 204, 0],
        [14, 0, 204],
        [0, 204, 160],
        [128, 200, 255],
        [255, 0, 109],
        [0, 255, 36],
        [189, 0, 204],
        [255, 0, 218],
        [0, 160, 204],
        [0, 255, 145],
        [204, 0, 131],
        [182, 0, 255],
        [255, 109, 0],
        [0, 255, 255],
        [72, 0, 255],
        [204, 43, 0],
        [204, 131, 0],
        [255, 0, 0],
        [72, 255, 0],
        [189, 204, 0],
        [182, 255, 0],
        [102, 0, 204],
        [32, 72, 204],
        [0, 145, 255],
        [14, 204, 0],
        [0, 128, 72],
        [204, 0, 43],
        [235, 205, 119],
        [115, 227, 112],
        [157, 113, 143],
        [132, 93, 50],
        [82, 21, 114],
    ]
    
    palette_length = len(ORIGINAL_GOLIATH_PALETTE)
    
    if num <= palette_length:
        return ORIGINAL_GOLIATH_PALETTE[:num]
    else:
        # Generate additional colors if num > palette_length
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Number of additional colors needed
        extra_num = num - palette_length
        
        # Generate colors using a colormap
        cmap = plt.get_cmap('hsv', extra_num)
        extra_colors = []
        
        for i in range(extra_num):
            color = cmap(i)
            # Convert from RGBA [0,1] to RGB [0,255]
            rgb = [int(255 * color[0]), int(255 * color[1]), int(255 * color[2])]
            extra_colors.append(rgb)
        
        return ORIGINAL_GOLIATH_PALETTE + extra_colors
    
