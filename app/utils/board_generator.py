import cv2.aruco as aruco
import numpy as np
from reportlab.lib.pagesizes import landscape, portrait, A4
from reportlab.pdfgen import canvas
from PIL import Image
import cv2 

# Step 1: Set parameters
board_shape = (7, 10)  # Number of squares (columns, rows)
square_size = 0.025  # Size of a square in meters
marker_size = 0.020  # Size of the ArUco marker in meters

# Convert square size to inches for the footer
square_size_in_mm = square_size * 1000  # Convert meters to millimeters

# Set DPI for high-quality print
DPI = 72

# Calculate physical board dimensions
board_width_inches = board_shape[0] * square_size * 39.37  # Convert meters to inches
board_height_inches = board_shape[1] * square_size * 39.37  # Convert meters to inches

# Calculate image dimensions in pixels for the specified DPI
board_width_px = int(board_width_inches * DPI)
board_height_px = int(board_height_inches * DPI)

# Step 2: Create Charuco board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
board = aruco.CharucoBoard(board_shape, square_size, marker_size, dictionary)

# Generate the board image
board_image = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

# Save the image as a temporary PNG file
temp_image_filename = "charuco_board_temp.png"
cv2.imwrite(temp_image_filename, board_image)

# Step 3: Determine page orientation
if board_shape[0] > board_shape[1]:
    page_orientation = landscape(A4)
else:
    page_orientation = portrait(A4)

# Page dimensions
page_width, page_height = page_orientation

# Center the image on the page
x_offset = (page_width - board_width_px * (72 / DPI)) / 2  # Convert from pixels to points
y_offset = (page_height - board_height_px * (72 / DPI)) / 2  # Convert from pixels to points

# Step 4: Create the PDF
pdf_filename = "charuco_board_with_info.pdf"
c = canvas.Canvas(pdf_filename, pagesize=page_orientation)

# Add the image to the PDF, centered
c.drawImage(temp_image_filename, x_offset, y_offset,
            width=board_width_px * (72 / DPI), height=board_height_px * (72 / DPI))

# Add footer with board information

footer_text = f"Charuco Board: {board_shape[0]} x {board_shape[1]} squares, " 
footer_text += f"Square Size: {square_size_in_mm:.2f} mm, "
footer_text += f"Marker Size: {marker_size * 1000:.2f} mm"
c.setFont("Helvetica", 10)
c.drawCentredString(page_width / 2, 20, footer_text)  # Place footer near the bottom

# Save the PDF
c.save()

# Clean up temporary image file
import os
# os.remove(temp_image_filename)

print(f"PDF saved as {pdf_filename}. Ensure you print it without scaling for accurate dimensions.")
