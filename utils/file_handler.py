import os
from PySide6.QtWidgets import QFileDialog

class FileHandler:
    def __init__(self, parent_widget):
        self.parent = parent_widget

    def open_file_dialog(self):
        """Opens Windows Explorer and returns the selected file path."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select a File to Upload",
            "",
            "All Files (*.*);;Images (*.png *.jpg *.jpeg);;Documents (*.txt *.pdf *.docx)"
        )
        return file_name

    def process_file(self, file_path):
        """
        Extracts metadata from the file to simulate AI analysis.
        """
        if not file_path:
            return None

        # Extract info
        name = os.path.basename(file_path)
        # Split extension (e.g., .png)
        extension = os.path.splitext(name)[1].lower()
        size_bytes = os.path.getsize(file_path)

        # Convert size to readable format
        if size_bytes < 1024:
            size_str = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

        return {
            "path": file_path,
            "name": name,
            "extension": extension,
            "size": size_str
        }