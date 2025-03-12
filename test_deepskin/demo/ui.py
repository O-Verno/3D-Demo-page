from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar
from PySide6.QtCore import QThread, Signal

import subprocess
import sys
import os

from src.wound_image import WoundImage


class Worker(QThread):
    """Threaded worker"""
    # Define signals to communicate back to the main thread
    finished = Signal()
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, folder_input: str, folder_output: str, logging: bool):
        super().__init__()
        self.folder_input = folder_input
        self.folder_output = folder_output
        self.logging = logging

    def run(self):
        try:
            # List all image files in the input folder
            wsis: list[WoundImage] = [
                WoundImage(
                    image_path=os.path.join(
                        self.folder_input,
                        file),
                    logging=self.logging)
                for file in os.listdir(self.folder_input)
                if file.endswith((".png", ".jpg", ".jpeg"))
            ]

            total_files = len(wsis)
            wounds_output_dir = os.path.join(self.folder_output, "wounds")
            csv_output_file = os.path.join(
                self.folder_output, "csv", "pwat_data.csv")

            for index, wi in enumerate(wsis):
                # Update progress bar
                progress = int(((index + 1) / total_files) * 100)
                self.progress.emit(progress)

                current_dir = os.path.join(
                    wounds_output_dir, os.path.basename(wi.image_path).replace(".", "_"))
                extension = "." + wi.image_path.split(".")[-1]

                # Save all data in the 'img_output_dir'
                wi.save_all(
                    img_output_dir=current_dir,
                    csv_output_file=csv_output_file,
                    file_extension=extension)

            if self.folder_output:
                # For Windows
                if os.name == 'nt':
                    os.startfile(self.folder_output)
                # For macOS
                elif sys.platform == 'darwin':
                    subprocess.run(['open', self.folder_output])
                # For Linux
                elif sys.platform == 'linux':
                    subprocess.run(['xdg-open', self.folder_output])

            self.finished.emit()  # Emit signal when done

        except Exception as e:
            self.error.emit(str(e))  # Emit signal if there is an error


class UI(QWidget):
    def __init__(self, logging: bool):
        super().__init__()

        self.logging = logging

        self.setWindowTitle("UI")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.folder_input = os.path.abspath(os.path.join("input"))
        self.folder_output = os.path.abspath(
            os.path.join("output", "demo", "ui"))

        self.label_input = QLabel(f"Input Folder: {self.folder_input}")
        self.label_output = QLabel(f"Output Folder: {self.folder_output}")
        self.label_generate = QLabel(f"Generate the predicted images")

        self.btn_input = QPushButton("Choose Input Folder")
        self.btn_input.clicked.connect(self.choose_folder_input)

        self.btn_output = QPushButton("Choose Output Folder")
        self.btn_output.clicked.connect(self.choose_folder_output)

        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.generate)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Range from 0 to 100
        self.progress_bar.setValue(0)  # Initial value is 0
        self.progress_bar.setVisible(False)  # Hide the progress bar initially

        layout.addWidget(self.label_input)
        layout.addWidget(self.btn_input)
        layout.addWidget(self.label_output)
        layout.addWidget(self.btn_output)
        layout.addWidget(self.label_generate)
        layout.addWidget(self.btn_generate)
        # Add the progress bar to the layout
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def choose_folder_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.folder_input = folder.replace("/", "\\")
            self.label_input.setText(f"Input Folder: {self.folder_input}")

    def choose_folder_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_output = folder.replace("/", "\\")
            self.label_output.setText(f"Output Folder: {self.folder_output}")

    def generate(self):
        if not self.folder_input or not self.folder_output:
            QMessageBox.warning(
                self, "Error", "Please select both folders before generating.")
        else:
            self.generation_finished_called = False

            self.btn_input.setEnabled(False)
            self.btn_output.setEnabled(False)
            self.btn_generate.setEnabled(False)
            self.progress_bar.setVisible(True)  # Show the progress bar

            print("Generating with:")
            print(f"  - Input Folder: {self.folder_input}")
            print(f"  - Output Folder: {self.folder_output}")

            # Start the worker thread to run the generate process
            self.worker = Worker(
                self.folder_input,
                self.folder_output,
                self.logging)

            # Connect the worker's signals
            self.worker.finished.connect(self.on_generation_finished)
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(self.on_progress_update)

            # Start the worker thread
            self.worker.start()

    def on_generation_finished(self):
        if self.generation_finished_called is False:
            self.generation_finished_called = True
            self.label_generate.setText(f"Generation done")
            self.reset_ui()

    def on_generation_error(self, error_message: str):
        QMessageBox.critical(
            self, "Error", f"An error occurred: {error_message}")
        self.reset_ui()

    def on_progress_update(self, value: int):
        self.progress_bar.setValue(value)  # Update the progress bar

    def reset_ui(self):
        self.btn_input.setEnabled(True)
        self.btn_output.setEnabled(True)
        self.btn_generate.setEnabled(True)
        # Hide the progress bar after generation
        self.progress_bar.setVisible(False)


def main():
    logging = True
    app = QApplication(sys.argv)
    window = UI(logging)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
