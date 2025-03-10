import sys
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel,
                               QPushButton, QHBoxLayout, QMessageBox, QSizePolicy)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class ResizableImageLabel(QLabel):
    """Custom QLabel that automatically resizes the image to fit the widget while maintaining aspect ratio."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)  # Prevent the label from becoming too small
        self.original_pixmap = None
        self.setScaledContents(False)  # We'll handle scaling manually

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        if self.original_pixmap:
            self.update_scaled_pixmap()

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)


class ImageConfirmationDialog(QDialog):
    def __init__(self, image_path, notice_text, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Confirm Action")

        self.image_path = image_path
        self.notice_text = notice_text
        self.original_pixmap = QPixmap(self.image_path) # Store original for aspect ratio

        if self.original_pixmap.isNull():
            QMessageBox.critical(self, "Error", f"Could not load image: {self.image_path}")
            self.reject()
            return

        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)

        # Image Label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.update_pixmap()  # Initial pixmap setup
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
        self.update_pixmap()
        super().resizeEvent(event)

    def update_pixmap(self):
        # Get the current size of the label
        label_size = self.image_label.size()

        # Calculate the scaled size while preserving aspect ratio
        original_size = self.original_pixmap.size()
        scaled_size = original_size.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio)

        # Create a scaled pixmap
        scaled_pixmap = self.original_pixmap.scaled(scaled_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Set the pixmap on the label
        self.image_label.setPixmap(scaled_pixmap)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     dialog = ImageConfirmationDialog("image.jpg", "Are you sure you want to proceed?")
#     dialog.exec()
#     sys.exit(app.exec())