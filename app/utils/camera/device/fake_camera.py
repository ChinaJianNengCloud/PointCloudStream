import numpy as np
import cv2
import logging
logger = logging.getLogger(__name__)

class FakeRGBDFrame:
    """Fake RGBD frame containing synthetic depth and color images."""
    def __init__(self, depth_image, color_image):
        self.depth = depth_image
        self.color = color_image

        
class FakeCamera:
    """Fake camera that generates synthetic RGBD frames for debugging."""
    def __init__(self):
        self.width = 640
        self.height = 480
        self.frame_idx = 0

    def connect(self, index):
        """Fake connect method, always returns True."""
        return True

    def release(self):
        pass

    def disconnect(self):
        """Fake disconnect method."""
        pass

    def read(self, both: bool = False):
        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        center_x = int((self.frame_idx * 5) % self.width)
        center_y = self.height // 2
        cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)

        # Generate a dynamic depth image
        x = np.linspace(0, 2 * np.pi, self.width)
        y = np.linspace(0, 2 * np.pi, self.height)
        xx, yy = np.meshgrid(x, y)

        # Use a sine wave to create dynamic depth variations
        base_depth = 1000  # Base depth value in mm
        amplitude = 500  # Amplitude of the depth variation
        depth_image = base_depth + amplitude * np.sin(xx + self.frame_idx * 0.1)

        # Convert to uint16
        depth_image = depth_image.astype(np.uint16)

        # Randomly zero out some regions to simulate missing depth
        num_missing_regions = np.random.randint(5, 15)  # Random number of missing regions
        for _ in range(num_missing_regions):
            # Randomly choose the size and position of the missing region
            start_x = np.random.randint(0, self.width - 50)
            start_y = np.random.randint(0, self.height - 50)
            width = np.random.randint(20, 100)
            height = np.random.randint(20, 100)

            # Zero out the region
            depth_image[start_y:start_y + height, start_x:start_x + width] = 0
        self.frame_idx += 1
        if not both:
            return True, color_image
        return True, FakeRGBDFrame(depth_image, color_image)

    def capture_frame(self, enable_align_depth_to_color=True):
        """Generate synthetic depth and color images with dynamic depth regions."""
        return self.read(both=True)[1]