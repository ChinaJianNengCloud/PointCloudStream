import subprocess
from typing import List, Dict

class USBVideoManager:
    """Manages video devices and operations."""
    def __init__(self):
        self.devices: List[Dict] = None
        self.find_video_devices()

    def get_id_by_name(self, name: str) -> int:
        for device in self.devices:
            if device['name'] == name:
                return device['id']
        return -1

    def get_name_by_id(self, id: int) -> str:
        for device in self.devices:
            if device['id'] == id:
                return device['name']
        return 'Unknown'

    def find_video_devices(self):
        """Get a list of available video devices."""
        self.devices = []
        name_count = {}  # Track counts of each device name

        try:
            # Run the v4l2-ctl command to list devices
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                    capture_output=True, text=True)

            lines = result.stdout.strip().split('\n')

            current_device_name = None
            for line in lines:
                if not line.startswith('\t'):
                    # This is a device name line
                    current_device_name = line.strip().rstrip(':')
                elif line.strip().startswith('/dev/video'):
                    device_path = line.strip()
                    has_formats = self.check_device_formats(device_path)

                    if has_formats:
                        base_name = current_device_name.split(':')[0].strip()

                        # Increment and tag device name if duplicate
                        count = name_count.get(base_name, 0)
                        name_count[base_name] = count + 1

                        if count == 0:
                            display_name = base_name
                        else:
                            display_name = f"{base_name} ({count})"

                        self.devices.append({
                            'name': display_name,
                            'path': device_path,
                            'id': int(device_path.replace('/dev/video', ''))
                        })

            return self.devices

        except Exception as e:
            print(f"Error getting video devices: {e}")
            return []

    def check_device_formats(self, device_path):
        """Check if a device has any supported formats."""
        try:
            # Run v4l2-ctl to check formats
            result = subprocess.run(
                ['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
                capture_output=True, text=True
            )
            
            # A valid device with formats will have lines containing "Size:"
            output = result.stdout
            return "Size:" in output
        except Exception as e:
            print(f"Error checking formats for {device_path}: {e}")
            return False
