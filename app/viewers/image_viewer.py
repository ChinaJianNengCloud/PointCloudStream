from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


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