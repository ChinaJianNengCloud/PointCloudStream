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
    

import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def create_board(board_shape=(11, 6), marker_size=0.018):

    square_size=8*marker_size/6

    print(f"bord_shape: {board_shape}, square_size: {square_size}, marker_size: {marker_size}")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    board = aruco.CharucoBoard(board_shape, square_size, marker_size, dictionary)
    
    dpi = 300  # Set to 300 DPI for high-quality print
    
    # Calculate image dimensions in mm
    image_width_mm = square_size * board_shape[0] * 1000  # Convert meters to mm
    image_height_mm = square_size * board_shape[1] * 1000  # Convert meters to mm

    # Convert mm dimensions to pixels using DPI
    image_width_px = int(round((image_width_mm * dpi) / 25.4))
    image_height_px = int(round((image_height_mm * dpi) / 25.4))
    
    # Generate the board image with specific pixel dimensions
    board_image = board.generateImage((image_width_px, image_height_px), marginSize=0, borderBits=1)
    cv2.imwrite("board_image.png", board_image)
    
    # Convert to RGB and rotate 90 degrees for saving with matplotlib
    board_image_rgb = cv2.cvtColor(board_image, cv2.COLOR_GRAY2RGB)
    board_image_rgb_rotated = np.rot90(board_image_rgb, k=1)  # Rotate 90 degrees
    board_image_rgb_rotated = np.flipud(board_image_rgb_rotated)  # Flip vertically
    # Since the image is rotated, swap width and height
    rotated_image_width_mm = image_height_mm
    rotated_image_height_mm = image_width_mm

    # Save as A4 PDF with correct DPI and additional information
    a4_width_mm = 210
    a4_height_mm = 297

    fig_width_in = a4_width_mm / 25.4
    fig_height_in = a4_height_mm / 25.4

    # Create a figure that matches the A4 size
    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, a4_width_mm)
    ax.set_ylim(0, a4_height_mm)
    ax.invert_yaxis()  # Invert y-axis to match PDF coordinate system

    # Calculate positions to center the rotated image on the A4 page
    left = (a4_width_mm - rotated_image_width_mm) / 2
    bottom = (a4_height_mm - rotated_image_height_mm) / 2

    # Insert the rotated image into the A4 page at the correct position and size
    extent = [left, left + rotated_image_width_mm, bottom, bottom + rotated_image_height_mm]
    ax.imshow(board_image_rgb_rotated, extent=extent)

    # Adding text information below the image
    info_text = f"Board Shape: {board_shape}\nSquare Size: {square_size:.3f} m Marker Size: {marker_size} m"
    text_y_position = bottom - 9  # 15 mm below the image

    fig.text(0.5, text_y_position / a4_height_mm, info_text, ha="center", fontsize=12)

    # Save the figure as a PDF with the correct DPI
    with PdfPages("BoardImage.pdf") as pdf:
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    print("An A4-sized PDF file named BoardImage.pdf with rotated board image and board information is generated in the folder containing this file.")


if __name__ == "__main__":
    create_board()