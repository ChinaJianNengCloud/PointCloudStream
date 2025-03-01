from PySide6.QtCore import QTimer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.entry import PCDStreamer

def on_stream_init_button_clicked(self: "PCDStreamer"):
    if self.streaming:
        # Stop streaming
        self.streaming = False
        self.stream_init_button.setText("Start")
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self.streamer, 'camera'):
            self.streamer.camera.disconnect()
        self.status_message.setText("System: Stream Stopped")
    else:
        # Start streaming
        self.streaming = True
        self.stream_init_button.setText("Stop")
        connected = self.streamer.camera_mode_init()
        if connected:
            self.status_message.setText("System: Streaming from Camera")
            self.timer = QTimer()
            self.timer.timeout.connect(self.frame_calling)
            self.timer.start()  # Update at ~33 FPS
        else:
            self.status_message.setText("System: Failed to connect to Camera")
        self.set_vtk_camera_from_intrinsics(self.streamer.intrinsic_matrix, self.streamer.extrinsics)
    self.set_enable_after_stream_init()