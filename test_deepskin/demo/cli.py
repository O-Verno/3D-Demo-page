import subprocess
import sys
import os

from src.wound_image import WoundImage


class CLI:
    """Non-Threaded CLI"""

    def __init__(self, logging: bool):
        self.logging = logging
        self.folder_input = None
        self.folder_output = None

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

            wounds_output_dir = os.path.join(self.folder_output, "wounds")
            csv_output_file = os.path.join(
                self.folder_output, "csv", "pwat_data.csv")

            for wi in wsis:
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

            print("Generation done")

        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    logging = True
    cli = CLI(logging)
    cli.folder_input = os.path.abspath(os.path.join("input"))
    cli.folder_output = os.path.abspath(os.path.join("output", "demo", "cli"))
    cli.run()


if __name__ == "__main__":
    main()
