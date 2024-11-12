
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