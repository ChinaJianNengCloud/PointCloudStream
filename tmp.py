import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette, QColor

class ThemeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Theme Detector")
        self.resize(400, 200)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create labels
        self.theme_label = QLabel("Checking theme...")
        self.theme_label.setAlignment(Qt.AlignCenter)
        self.theme_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        self.bg_color_label = QLabel("Background color: ")
        self.bg_color_label.setAlignment(Qt.AlignCenter)
        
        self.text_color_label = QLabel("Text color: ")
        self.text_color_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(self.theme_label)
        layout.addWidget(self.bg_color_label)
        layout.addWidget(self.text_color_label)
        
        # Set up a timer to periodically check the theme (could also use an event handler)
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_theme)
        self.timer.start(1000)  # Check theme every second
        
        # Initial theme check
        self.check_theme()
    
    def check_theme(self):
        # Get the application palette
        palette = QApplication.palette()
        
        # Get background and text colors
        bg_color = palette.color(QPalette.Window)
        text_color = palette.color(QPalette.WindowText)
        
        # Calculate luminance (0.299*R + 0.587*G + 0.114*B)
        # This is a common formula to determine perceived brightness
        luminance = (0.299 * bg_color.red() + 
                     0.587 * bg_color.green() + 
                     0.114 * bg_color.blue()) / 255
        
        # If luminance is less than 0.5, it's considered dark mode
        is_dark_mode = luminance < 0.5
        
        # Update UI
        theme_text = "Dark Mode" if is_dark_mode else "Light Mode"
        self.theme_label.setText(f"Current Theme: {theme_text}")
        
        # Show color values
        self.bg_color_label.setText(f"Background color: RGB({bg_color.red()}, {bg_color.green()}, {bg_color.blue()})")
        self.text_color_label.setText(f"Text color: RGB({text_color.red()}, {text_color.green()}, {text_color.blue()})")
        
        # Optional: Set a border around the window to visualize the theme better
        border_color = "white" if is_dark_mode else "black"
        self.setStyleSheet(f"QMainWindow {{ border: 2px solid {border_color}; }}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Alternative method to check if dark mode is enabled:
    # This checks if the application style hints the app is in dark mode
    # Only works on some platforms and with newer Qt versions
    is_dark_theme = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    print(f"Qt styleHints reports dark theme: {is_dark_theme}")
    
    window = ThemeDetectorApp()
    window.show()
    
    sys.exit(app.exec())