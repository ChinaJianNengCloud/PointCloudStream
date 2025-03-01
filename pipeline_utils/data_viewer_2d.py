import sys
import os
import json
import csv  # Import the CSV module

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QComboBox, QFileDialog  # Import QComboBox and QFileDialog
)
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from PySide6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Files Checker")

        # Define folders (assumes a structure: data folder with many subfolders and a "resources" folder)
        self.data_folder = "/home/capre/disk_4/yutao/data"  # Replace with your actual path
        self.resources_folder = os.path.join(self.data_folder, "resources")

        # Global list of all items from all JSON files
        self.items = []
        # Current global index
        self.index_in_all = 0
        # Dictionary to keep marks, key is (folder, uuid)
        self.marks = {}

        # Cache for the currently loaded QPixmaps for the first and last images.
        self.current_first_pixmap = None
        self.current_last_pixmap = None

        # List of folders to display in the combobox
        self.folder_list = []

        # Load all JSON items from subfolders (skipping "resources")
        self.load_data()

        # Build the UI
        self.init_ui()
        self.update_display()

    def load_data(self):
        """Traverse subdirectories (except resources) and load JSON items."""
        self.folder_list = []  # Reset the folder list before loading data
        self.items = []  # Reset the items list
        for folder in os.listdir(self.data_folder):
            full_path = os.path.join(self.data_folder, folder)
            if os.path.isdir(full_path) and folder != "resources":
                self.folder_list.append(folder)  # Add folder to list
                for file in os.listdir(full_path):
                    if file.endswith("saved_data.json"):
                        json_path = os.path.join(full_path, file)
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        except Exception as e:
                            print(f"Failed to load {json_path}: {e}")
                            continue
                        items_in_json = []
                        for uuid, info in data.items():
                            item = {
                                "uuid": uuid,
                                "prompt": info.get("prompt", ""),
                                "color_files": info.get("color_files", []),
                                "folder": folder,  # parent folder name
                            }
                            items_in_json.append(item)
                        for i, item in enumerate(items_in_json):
                            item["local_index"] = i + 1   # starting at 1 for display
                            item["local_count"] = len(items_in_json)
                        self.items.extend(items_in_json)
        self.total_count = len(self.items)

    def init_ui(self):
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Folder selection combobox
        self.folder_combo = QComboBox()
        self.folder_combo.addItems(self.folder_list)
        self.folder_combo.currentIndexChanged.connect(self.on_folder_changed)
        layout.addWidget(self.folder_combo)

        # Info label to display details about the current item.
        self.info_label = QLabel("Info: ")
        layout.addWidget(self.info_label)

        # Layout for the two images (first and last color file)
        img_layout = QHBoxLayout()
        self.image_label_first = QLabel("First image")
        self.image_label_last = QLabel("Last image")
        self.image_label_first.setAlignment(Qt.AlignCenter)
        self.image_label_last.setAlignment(Qt.AlignCenter)
        # Make the image labels expand to fill available space.
        self.image_label_first.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label_last.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout.addWidget(self.image_label_first)
        img_layout.addWidget(self.image_label_last)
        layout.addLayout(img_layout)

        # Button layout: Navigation and marking controls
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_valid = QPushButton("Valid")
        self.btn_invalid = QPushButton("Invalid")
        self.btn_save = QPushButton("Save Marks")
        self.btn_load = QPushButton("Load Marks")
        self.btn_prev_valid = QPushButton("Prev Valid")  # New button
        self.btn_next_valid = QPushButton("Next Valid")  # New button
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_prev_valid)
        btn_layout.addWidget(self.btn_next_valid)
        btn_layout.addWidget(self.btn_valid)
        btn_layout.addWidget(self.btn_invalid)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)

        # Connect button signals.
        self.btn_prev.clicked.connect(self.prev_item)
        self.btn_next.clicked.connect(self.next_item)
        self.btn_valid.clicked.connect(self.mark_valid)
        self.btn_invalid.clicked.connect(self.mark_invalid)
        self.btn_save.clicked.connect(self.save_marks)
        self.btn_load.clicked.connect(self.load_marks)
        self.btn_prev_valid.clicked.connect(self.prev_valid)  # Connect new button
        self.btn_next_valid.clicked.connect(self.next_valid)  # Connect new button

        # Add key bindings (Ctrl+Q and Ctrl+E for previous/next valid).
        self.shortcut_prev_valid = self.add_shortcut("Q", self.prev_valid)
        self.shortcut_next_valid = self.add_shortcut("E", self.next_valid)
        self.shortcut_prev = self.add_shortcut("A", self.prev_item)
        self.shortcut_next = self.add_shortcut("D", self.next_item)
        self.shortcut_invalid = self.add_shortcut("C", self.mark_invalid)

    def add_shortcut(self, key, callback):
        shortcut = QKeySequence(key)
        return self.add_shortcut_for_sequence(shortcut, callback)

    def add_shortcut_for_sequence(self, sequence, callback):
        shortcut =  self.add_shortcut_helper(sequence, callback)
        shortcut.setAutoRepeat(False)
        return shortcut

    def add_shortcut_helper(self, sequence, callback):
        shortcut = QShortcut(sequence, self) #corrected line
        shortcut.activated.connect(callback)
        return shortcut

    def update_display(self):
        """Update info label and image display for the current item."""
        if not self.items:
            self.info_label.setText("No items loaded")
            self.image_label_first.clear()
            self.image_label_last.clear()
            return

        current_item = self.items[self.index_in_all]
        folder = current_item["folder"]
        prompt = current_item["prompt"]
        local_index = current_item.get("local_index", 1)
        local_count = current_item.get("local_count", 1)

        # Count how many items are marked as invalid.
        invalid_count = sum(1 for mark in self.marks.values() if mark == "Invalid")

        info_text = (
            f"Folder: {folder} | Prompt: {prompt} | "
            f"JSON item: {local_index}/{local_count} | "
            f"Global: {self.index_in_all+1}/{self.total_count} | "
            f"Invalid Count: {invalid_count}"
        )
        key = (folder, current_item["uuid"])
        mark = self.marks.get(key, "Unmarked")
        info_text += f" | Mark: {mark}"
        self.info_label.setText(info_text)

        # Load the first and last images from the color_files list.
        color_files = current_item["color_files"]
        self.current_first_pixmap = None
        self.current_last_pixmap = None
        self.image_label_first.setText("First image") # Reset the label
        self.image_label_last.setText("Last image")  # Reset the label

        if color_files:
            first_img_path = os.path.join(self.resources_folder, color_files[0])
            last_img_path = os.path.join(self.resources_folder, color_files[-1])
            if os.path.exists(first_img_path):
                self.current_first_pixmap = QPixmap(first_img_path)
            else:
                self.image_label_first.setText("First image not found")
            if os.path.exists(last_img_path):
                self.current_last_pixmap = QPixmap(last_img_path)
            else:
                self.image_label_last.setText("Last image not found")
        else:
            self.image_label_first.setText("No color files")
            self.image_label_last.setText("No color files")
        # Update image labels (scaling them to fit the label size).
        self.update_images()

    def update_images(self):
        """Scale and set the pixmaps based on the current size of the image labels."""
        if self.current_first_pixmap:
            scaled_first = self.current_first_pixmap.scaled(
                self.image_label_first.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label_first.setPixmap(scaled_first)
        if self.current_last_pixmap:
            scaled_last = self.current_last_pixmap.scaled(
                self.image_label_last.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label_last.setPixmap(scaled_last)

    def resizeEvent(self, event):
        """Handle window resize to adjust images."""
        super().resizeEvent(event)
        self.update_images()

    def next_item(self):
        """Show the next item, if available."""
        if self.index_in_all < self.total_count - 1:
            self.index_in_all += 1
            self.update_display()

    def prev_item(self):
        """Show the previous item, if available."""
        if self.index_in_all > 0:
            self.index_in_all -= 1
            self.update_display()

    def mark_valid(self):
        """Mark the current item as valid."""
        if not self.items:
            return
        current_item = self.items[self.index_in_all]
        key = (current_item["folder"], current_item["uuid"])
        self.marks[key] = "Valid"
        self.update_display()

    def mark_invalid(self):
        """Mark the current item as invalid."""
        if not self.items:
            return
        current_item = self.items[self.index_in_all]
        key = (current_item["folder"], current_item["uuid"])
        self.marks[key] = "Invalid"
        self.update_display()

    def save_marks(self):
        """Save marks to a CSV file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Marks", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['folder_name', 'uuid', 'mark'])  # Header row
                    for (folder, uuid), mark in self.marks.items():
                        csv_writer.writerow([folder, uuid, mark])
                print("Marks saved to CSV.")
            except Exception as e:
                print("Error saving marks:", e)

    def load_marks(self):
        """Load marks from a CSV file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Marks", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader, None)  # Skip header row
                    self.marks = {}
                    for row in csv_reader:
                        if len(row) == 3:
                            folder, uuid, mark = row
                            self.marks[(folder, uuid)] = mark
                self.update_display()
                print("Marks loaded from CSV.")
            except Exception as e:
                print("Error loading marks:", e)

    def is_valid(self, index):
        """Check if the item at the given index is considered valid."""
        if not self.items or index < 0 or index >= len(self.items):
            return False

        item = self.items[index]
        key = (item["folder"], item["uuid"])
        mark = self.marks.get(key, "Unmarked")
        return mark != "Invalid"

    def next_valid(self):
        """Move to the next item that is considered 'valid'."""
        if not self.items:
            return

        start_index = self.index_in_all
        next_index = (start_index + 1) % len(self.items)  # Wrap around
        while next_index != start_index:
            if self.is_valid(next_index):
                self.index_in_all = next_index
                self.update_display()
                return
            next_index = (next_index + 1) % len(self.items)  # Increment and wrap

        # If we reach the start index without finding a valid item, stay where we are.
        if self.is_valid(start_index): # we have one to show, which is the current one
            self.update_display()

    def prev_valid(self):
        """Move to the previous item that is considered 'valid'."""
        if not self.items:
            return

        start_index = self.index_in_all
        prev_index = (start_index - 1 + len(self.items)) % len(self.items)  # Wrap around
        while prev_index != start_index:
            if self.is_valid(prev_index):
                self.index_in_all = prev_index
                self.update_display()
                return
            prev_index = (prev_index - 1 + len(self.items)) % len(self.items)  # Decrement and wrap

        # If we reach the start index without finding a valid item, stay where we are.
        if self.is_valid(start_index): # we have one to show, which is the current one
            self.update_display()

    def keyPressEvent(self, event):
        """Add key bindings: A for previous, D for next, C for invalid mark."""
        key = event.key()
        if key == Qt.Key_A:
            self.prev_item()
        elif key == Qt.Key_D:
            self.next_item()
        elif key == Qt.Key_C:
            self.mark_invalid()
        elif key == Qt.Key_Q:
            self.prev_valid()
        elif key == Qt.Key_E:
            self.next_valid()
        else:
            super().keyPressEvent(event)

    def on_folder_changed(self, index):
        """Handle folder selection change in the combobox."""
        selected_folder = self.folder_combo.itemText(index)
        self.filter_items_by_folder(selected_folder)
        self.index_in_all = 0  # Reset to the beginning of the filtered list
        self.update_display()

    def filter_items_by_folder(self, folder):
        """Filter the items list to show only items from the selected folder."""
        if folder == "All":
            self.load_data()
        else:
            self.items = [item for item in self.items if item["folder"] == folder]
        self.total_count = len(self.items)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())