import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from PySide6.QtGui import QPalette
import pyqtgraph as pg

class RobotPoseGraphWidget(QWidget):
    def __init__(self, max_points=300):
        super().__init__()
        self.max_points = max_points
        self.current_state_type = "tcp"  # Default state type
        self.plots = {}
        self.channel_data = {}
        self.reset_data("tcp")  # Initialize with TCP data structure
        self.data_index = 0
        
        # Theme-related variables
        self.is_dark_mode = self.detect_dark_mode()
        
        self.initUI()
        
        # Set up timer to check for theme changes
        self.theme_timer = QTimer()
        self.theme_timer.timeout.connect(self.check_theme)
        self.theme_timer.start(2000)  # Check every 2 seconds
        
    def detect_dark_mode(self):
        """Detect if the system is using dark mode"""
        palette = QApplication.palette()
        bg_color = palette.color(QPalette.Window)
        
        # Calculate luminance (0.299*R + 0.587*G + 0.114*B)
        luminance = (0.299 * bg_color.red() + 
                    0.587 * bg_color.green() + 
                    0.114 * bg_color.blue()) / 255
        
        # If luminance is less than 0.5, it's considered dark mode
        return luminance < 0.5
        
    def check_theme(self):
        """Check if the theme has changed and update if needed"""
        current_dark_mode = self.detect_dark_mode()
        if current_dark_mode != self.is_dark_mode:
            self.is_dark_mode = current_dark_mode
            self.apply_theme()
            self.setup_plots()  # Recreate plots with new theme
            
    def apply_theme(self):
        """Apply appropriate theming based on dark/light mode"""
        # Get the actual window background color
        palette = QApplication.palette()
        bg_color = palette.color(QPalette.Window)
        text_color = palette.color(QPalette.WindowText)
        
        # Convert Qt color to pyqtgraph format (RGB tuple)
        bg_rgb = (bg_color.red(), bg_color.green(), bg_color.blue())
        text_rgb = (text_color.red(), text_color.green(), text_color.blue())
        
        # Set graph background to match window background
        self.plot_widget.setBackground(bg_rgb)
        
        # Set axis colors to match text color
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=text_rgb, width=1))
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=text_rgb, width=1))
        
        # Text colors for axis labels
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=text_rgb, width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=text_rgb, width=1))
            
    def reset_data(self, state_type):
        """Reset data structures for the specified state type"""
        self.current_state_type = state_type
        self.data_index = 0
        
        # Clear existing data
        if state_type == "tcp":
            self.channel_data = {
                'x': np.zeros(self.max_points),
                'y': np.zeros(self.max_points),
                'z': np.zeros(self.max_points),
                'rx': np.zeros(self.max_points),
                'ry': np.zeros(self.max_points),
                'rz': np.zeros(self.max_points)
            }
        else:  # joint
            self.channel_data = {
                'joint_idx1': np.zeros(self.max_points),
                'joint_idx2': np.zeros(self.max_points),
                'joint_idx3': np.zeros(self.max_points),
                'joint_idx4': np.zeros(self.max_points),
                'joint_idx5': np.zeros(self.max_points),
                'joint_idx6': np.zeros(self.max_points)
            }
        
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        
        # Title label
        self.title_label = QLabel("Robot Pose (TCP)")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Create the plot widget with fixed height
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMaximumHeight(250)
        self.plot_widget.setMinimumHeight(250)
        layout.addWidget(self.plot_widget)
        
        # Apply theme based on system settings
        self.apply_theme()
        
        # Set labels
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time')
        
        # Create the legend and plots
        self.setup_plots()
        
    def setup_plots(self):
        """Set up the plots based on the current state type"""
        # Clear existing plots
        self.plot_widget.clear()
        self.plots = {}
        
        # Create a legend
        self.plot_widget.addLegend()
        
        # Create plot lines for each channel with different colors
        if self.current_state_type == "tcp":
            colors = {
                'x': (255, 0, 0), 
                'y': (0, 255, 0), 
                'z': (0, 0, 255), 
                'rx': (255, 128, 0), 
                'ry': (128, 0, 255), 
                'rz': (0, 128, 128)
            }
            self.title_label.setText("Robot Pose (TCP)")
        else:  # joint
            colors = {
                'joint_idx1': (255, 0, 0),
                'joint_idx2': (0, 255, 0),
                'joint_idx3': (0, 0, 255),
                'joint_idx4': (255, 128, 0),
                'joint_idx5': (128, 0, 255),
                'joint_idx6': (0, 128, 128)
            }
            self.title_label.setText("Robot Pose (Joint)")
        
        # Create plot lines
        for channel, color in colors.items():
            pen = pg.mkPen(color=color, width=1)
            self.plots[channel] = self.plot_widget.plot(
                self.channel_data[channel], 
                name=channel, 
                pen=pen
            )
    
    def set_state_type(self, state_type):
        """Change the state type and reset the plots"""
        if state_type != self.current_state_type:
            self.reset_data(state_type)
            self.setup_plots()
        
    def update_data(self, robot_pose, state_type="tcp"):
        """Update the plot data with new robot pose information"""
        if robot_pose is None or len(robot_pose) < 6:
            return
            
        # If state type has changed, reset the plots
        if state_type != self.current_state_type:
            self.set_state_type(state_type)
            
        # Update the data arrays with new values (circular buffer)
        if state_type == "tcp":
            channels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        else:  # joint
            channels = ['joint_idx1', 'joint_idx2', 'joint_idx3', 
                       'joint_idx4', 'joint_idx5', 'joint_idx6']
            
        for i, channel in enumerate(channels):
            if i < len(robot_pose):
                self.channel_data[channel][self.data_index] = robot_pose[i]
                
        # Update the data index for circular buffer
        self.data_index = (self.data_index + 1) % self.max_points
        
        # Update the plots with the new data
        for channel, plot in self.plots.items():
            # Reorder the data so that it's plotted correctly
            display_data = np.roll(self.channel_data[channel], -self.data_index)
            plot.setData(display_data)
