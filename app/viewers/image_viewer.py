from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                               QPushButton, QHBoxLayout, QSizePolicy)
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor, QPainterPath
from PySide6.QtCore import Qt, QRectF
import time

class ResizableImageLabel(QLabel):
    """Custom QLabel that automatically resizes the image to fit the widget while maintaining aspect ratio."""
    def __init__(self, parent=None, camera_name=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)  # Prevent the label from becoming too small
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.camera_name = camera_name
        self.setScaledContents(False)  # We'll handle scaling manually

    def set_camera_name(self, name):
        """Set the camera name to be displayed on the image"""
        self.camera_name = name
        self.update()

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        if self.original_pixmap:
            self.update_scaled_pixmap()

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        self.scaled_pixmap = self.original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        # Don't call super().setPixmap here, as we'll paint in paintEvent
        self.update()

    def paintEvent(self, event):
        if not self.scaled_pixmap:
            return super().paintEvent(event)
            
        # Calculate the position to center the image in the label
        x = (self.width() - self.scaled_pixmap.width()) // 2
        y = (self.height() - self.scaled_pixmap.height()) // 2
        
        painter = QPainter(self)
        # Draw the scaled image
        painter.drawPixmap(x, y, self.scaled_pixmap)
        
        # Draw the camera name if available
        if self.camera_name and self.scaled_pixmap:
            # Prepare font
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            
            # Measure text
            text_rect = painter.fontMetrics().boundingRect(self.camera_name)
            text_width = text_rect.width() + 16  # Add padding
            text_height = text_rect.height() + 8  # Add padding
            
            # Define background rectangle position and size (upper left corner)
            bg_rect = QRectF(x + 10, y + 10, text_width, text_height)
            
            # Create rounded rectangle path
            path = QPainterPath()
            path.addRoundedRect(bg_rect, 8, 8)  # 8px corner radius
            
            # Draw semi-transparent background
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillPath(path, QColor(0, 0, 0, 128))  # 50% opacity black
            
            # Draw text
            painter.setPen(QColor(255, 255, 255))  # White text for contrast
            painter.drawText(bg_rect, Qt.AlignCenter, self.camera_name)
        
        painter.end()


class ImageConfirmationDialog(QDialog):
    def __init__(self, image_path, notice_text, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Confirm Action")

        self.image_path = image_path
        self.notice_text = notice_text
        self.original_pixmap = None
        if image_path is not None:
            self.original_pixmap = QPixmap(self.image_path) # Store original for aspect ratio


        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)

        # Image Label
        self.image_label = ResizableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self.original_pixmap is not None:
            self.image_label.setPixmap(self.original_pixmap)  # Initial pixmap setup
        main_layout.addWidget(self.image_label)

        # Notice Text Label
        self.notice_label = QLabel(self.notice_text)
        self.notice_label.setWordWrap(True)
        self.notice_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.notice_label)

        # Button Layout (Confirm and Cancel)
        button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        button_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

        # Set Dialog Properties
        self.setLayout(main_layout)
        self.setMinimumWidth(300)

    def resizeEvent(self, event):
        # Override the resize event to update the pixmap when the dialog is resized
        if self.original_pixmap is not None:
            self.image_label.update_scaled_pixmap()
        super().resizeEvent(event)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     dialog = ImageConfirmationDialog("image.jpg", "Are you sure you want to proceed?")
#     dialog.exec()
#     sys.exit(app.exec())