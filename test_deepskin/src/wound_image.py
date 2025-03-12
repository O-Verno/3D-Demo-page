import re
import os
import csv
import cv2
import uuid
import shutil
import logging
import datetime
import pylab as plt
import tensorflow as tf

from numpy import ndarray
from typing import Optional
from deepskin.imgproc import imfill, get_perilesion_mask
from deepskin import wound_segmentation, evaluate_PWAT_score


from src.rgb import RGB


# Suppress TensorFlow logging messages
tf.get_logger().setLevel(logging.ERROR)


class WoundImage:
    """
    A class to process and analyze wound images.

    This class provides functionality to load, process, and visualize wound images.
    It includes methods for segmentation, masking, and PWAT (Pressure Ulcer Assessment Tool) scoring.

    Attributes:
        image_path (str): Path to the wound image file.
        logging (bool): Whether to enable logging for debugging purposes.
        _image (ndarray): The loaded image as a NumPy array.
        _segmentation (ndarray): The segmentation mask of the image.
        _wound_mask (ndarray): Mask for the wound area.
        _body_mask (ndarray): Mask for the body area.
        _bg_mask (ndarray): Mask for the background area.
        _wound_masked (ndarray): Image with only the wound area visible.
        _peri_wound_mask (ndarray): Mask for the peri-wound area.
        _peri_wound_masked (ndarray): Image with only the peri-wound area visible.
        _predicted_pwat (float): Predicted PWAT score.
        _clinical_pwat (float): Clinical PWAT score.
        _temp_dir (str): Directory for temporary files.
    """

    def __init__(self, image_path: str, logging: bool):
        """
        Initialize the WoundImage object.

        Args:
            image_path (str): Path to the wound image file.
            logging (bool): Whether to enable logging for debugging purposes.

        Raises:
            ValueError: If the image path is not a valid folder architecure or file format.
            FileNotFoundError: If any of the RGB values are outside the range 0-255.
        """
        self._valid_image_path(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File {image_path} not found.")

        self.image_path: str = image_path
        self.logging: bool = logging

        # Initialize attributes to None
        self._image: Optional[ndarray] = None
        self._segmentation: Optional[ndarray] = None
        self._wound_mask: Optional[ndarray] = None
        self._body_mask: Optional[ndarray] = None
        self._bg_mask: Optional[ndarray] = None
        self._wound_masked: Optional[ndarray] = None
        self._peri_wound_mask: Optional[ndarray] = None
        self._peri_wound_masked: Optional[ndarray] = None
        self._predicted_pwat: Optional[float] = None
        self._clinical_pwat: Optional[float] = None

        # Temporary directory for storing intermediate files
        self._temp_dir: str = os.path.join("output", "src")

    def log(self, msg: str):
        """
        Log a message if logging is enabled.

        Args:
            msg (str): The message to log.
        """
        if self.logging is True:
            print(msg)

    def show_all(self):
        """
        Display all processed images in a single plot.

        This method saves all intermediate images to a temporary directory,
        displays them in a grid, and then cleans up the temporary files.
        """
        current_dir = os.path.join(self._temp_dir, str(uuid.uuid4()))
        cof = os.path.join(self._temp_dir, "pwat_data.csv")
        fe = ".png"
        os.makedirs(self._temp_dir, exist_ok=True)
        self.log(f"Created {self._temp_dir}")
        self.save_all(
            img_output_dir=current_dir,
            csv_output_file=cof,
            file_extension=fe)

        def get_save_path(filename: str) -> str:
            return os.path.join(current_dir, filename + fe)

        files = [
            "original",
            "segmentation_mask",
            "segmentation_semantic",
            "mask_wound",
            "mask_peri_wound",
            "masked_wound",
            "masked_peri_wound",
            "pwat_estimation"
        ]
        fig, axes = plt.subplots(2, 4)
        fig.subplots_adjust(
            left=0,
            right=1,
            top=1,
            bottom=0,
            wspace=0,
            hspace=0)
        axes = axes.ravel()
        for i in range(len(files)):
            # Convert BGR to RGB
            img = cv2.imread(get_save_path(files[i]))[..., ::-1]
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(files[i])
        plt.gcf().canvas.manager.set_window_title("Plots")
        plt.show()
        plt.close()
        os.remove(cof)
        self.log(f"Removed {cof}")
        shutil.rmtree(current_dir)
        self.log(f"Removed {current_dir}")

    def show_original(self):
        """
        Display the original image.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_original(current_file)
        self._show_img(current_file, "original")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_segmentation_mask(self):
        """
        Display the segmentation mask.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_segmentation_mask(current_file)
        self._show_img(current_file, "segmentation_mask")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_segmentation_semantic(self):
        """
        Display the semantic segmentation with contours.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_segmentation_semantic(current_file)
        self._show_img(current_file, "segmentation_semantic")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_mask_wound(self):
        """
        Display the wound mask.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_mask_wound(current_file)
        self._show_img(current_file, "mask_wound")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_mask_peri_wound(self):
        """
        Display the peri-wound mask.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_mask_peri_wound(current_file)
        self._show_img(current_file, "mask_peri_wound")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_masked_wound(self):
        """
        Display the image with only the wound area visible.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_masked_wound(current_file)
        self._show_img(current_file, "masked_wound")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_masked_peri_wound(self):
        """
        Display the image with only the peri-wound area visible.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_masked_peri_wound(current_file)
        self._show_img(current_file, "masked_peri_wound")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def show_pwat_estimation(self):
        """
        Display the PWAT estimation overlay on the image.
        """
        current_file = os.path.join(self._temp_dir, str(uuid.uuid4()) + ".png")
        self.save_pwat_estimation(current_file)
        self._show_img(current_file, "pwat_estimation")
        os.remove(current_file)
        self.log(f"Removed {current_file}")

    def _show_img(self, img_path: str, title: str):
        """
        Display an image from the given path.

        Args:
            img_path (str): Path to the image file.
            title (str): Title for the displayed image.
        """
        img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
        plt.imshow(img)
        plt.title(title)
        plt.gcf().canvas.manager.set_window_title(title)
        plt.show()
        plt.close()

    def save_all(self, img_output_dir: str,
                 csv_output_file: str, file_extension: str):
        """
        Save all processed images and PWAT data to files.

        Args:
            img_output_dir (str): Directory to save image files.
            csv_output_file (str): Path to save PWAT data as a CSV file.
            file_extension (str): File extension for saved images (e.g., '.png').

        Raises:
            ValueError: If the file extension is not a valid image format.
        """
        if file_extension not in ('.png', '.jpg', '.jpeg'):
            raise ValueError(
                f"{file_extension} is not a valid file extension, try .pgn/.jpeg/.jpg instead.")

        def get_save_path(filename: str) -> str:
            return os.path.join(img_output_dir, filename + file_extension)

        self.save_original(get_save_path("original"))
        self.save_segmentation_mask(get_save_path("segmentation_mask"))
        self.save_segmentation_semantic(get_save_path("segmentation_semantic"))
        self.save_mask_wound(get_save_path("mask_wound"))
        self.save_mask_peri_wound(get_save_path("mask_peri_wound"))
        self.save_masked_wound(get_save_path("masked_wound"))
        self.save_masked_peri_wound(get_save_path("masked_peri_wound"))
        self.save_pwat_estimation(get_save_path("pwat_estimation"))
        self.save_pwat_to_csv(csv_output_file)

    def save_original(self, file_path: str):
        """
        Save the original image to a file.

        Args:
            file_path (str): Path to save the image.
        """
        self._save_img(file_path, self.get_image().copy())

    def save_segmentation_mask(self, file_path: str):
        """
        Save the segmentation mask to a file.

        Args:
            file_path (str): Path to save the mask.
        """
        self._save_img(file_path, self.get_segmentation().copy())

    def save_segmentation_semantic(self, file_path: str):
        """
        Save the semantic segmentation with contours to a file.

        Args:
            file_path (str): Path to save the image.
        """
        img = self.get_image().copy()
        contours_body, _ = cv2.findContours(
            self.get_body_mask().copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        contours_wound, _ = cv2.findContours(
            self.get_wound_mask().copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(img, contours_body, -1, RGB.BLUE, 2)
        cv2.drawContours(img, contours_wound, -1, RGB.GREEN, 2)
        self._save_img(file_path, img)

    def save_mask_wound(self, file_path: str):
        """
        Save the wound mask to a file.

        Args:
            file_path (str): Path to save the mask.
        """
        img = self.get_image().copy()
        contours_wound, _ = cv2.findContours(
            self.get_wound_mask().copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(img, contours_wound, -1, RGB.GREEN, 2)
        self._save_img(file_path, img)

    def save_mask_peri_wound(self, file_path: str):
        """
        Save the peri-wound mask to a file.

        Args:
            file_path (str): Path to save the mask.
        """
        img = self.get_image().copy()
        contours_peri_wound, _ = cv2.findContours(
            self.get_peri_wound_mask().copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(img, contours_peri_wound, -1, RGB.GREEN, 2)
        self._save_img(file_path, img)

    def save_masked_wound(self, file_path: str):
        """
        Save the image with only the wound area visible to a file.

        Args:
            file_path (str): Path to save the image.
        """
        self._save_img(file_path, self.get_wound_masked().copy())

    def save_masked_peri_wound(self, file_path: str):
        """
        Save the image with only the peri-wound area visible to a file.

        Args:
            file_path (str): Path to save the image.
        """
        self._save_img(file_path, self.get_peri_wound_masked().copy())

    def save_pwat_estimation(self, file_path: str):
        """
        Save the PWAT estimation overlay on the image to a file.

        Args:
            file_path (str): Path to save the image.
        """
        img = self.get_image().copy()
        contours_wound, _ = cv2.findContours(
            self.get_wound_mask().copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(img, contours_wound, -1, RGB.GREEN, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = RGB.BLACK
        border_color = RGB.WHITE
        line_type = cv2.LINE_AA
        x, y = 10, 30

        # Countour
        cv2.putText(
            img,
            "PWAT Estimation",
            (x, y),
            font,
            font_scale,
            border_color,
            font_thickness + 1,
            line_type)
        # Text
        cv2.putText(
            img,
            "PWAT Estimation",
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            line_type)
        y += 20
        pwat_clinical = self.get_clinical_pwat()

        if pwat_clinical > 0.0:
            # Countour
            cv2.putText(
                img,
                f"Clinical score: {pwat_clinical:.3f}",
                (x, y),
                font,
                font_scale,
                border_color,
                font_thickness + 1,
                line_type)
            # Text
            cv2.putText(
                img,
                f"Clinical score: {pwat_clinical:.3f}",
                (x, y),
                font,
                font_scale,
                text_color,
                font_thickness,
                line_type)
            y += 20
        # Countour
        cv2.putText(
            img,
            f"Predicted score: {self.get_predicted_pwat():.3f}",
            (x, y),
            font,
            font_scale,
            border_color,
            font_thickness + 1,
            line_type)
        # Text
        cv2.putText(
            img,
            f"Predicted score: {self.get_predicted_pwat():.3f}",
            (x, y),
            font,
            font_scale,
            text_color,
            font_thickness,
            line_type)
        self._save_img(file_path, img)

    def _save_img(self, file_path: str, bgr_img: ndarray):
        """
        Save an image to a file.

        Args:
            file_path (str): Path to save the image.
            bgr_img (ndarray): The image in BGR format.

        Raises:
            ValueError: If the image path is not a valid folder architecure or file format.
        """
        self._valid_image_path(file_path)
        dir_path = os.path.dirname(file_path)
        if dir_path:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                self.log(f"Created {dir_path}")
        rgb_img = bgr_img[..., ::-1]  # Convert BGR to RGB
        cv2.imwrite(file_path, rgb_img)
        self.log(f"Created {file_path}")

    def save_pwat_to_csv(self, file_path: str) -> None:
        """
        Save PWAT data to a CSV file.

        Args:
            file_path (str): Path to save the CSV file.

        Raises:
            ValueError: If the file path is not a valid folder architecure or .csv file format.
            ValueError: If the CSV header of an existing CSV not match the expected format.
        """
        pattern = r"^(?:[A-Za-z]:\\|/)?(?:[\w\s.-]+[/\\])*[\w\s.-]+\.csv$"
        match = re.match(pattern, file_path, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"File {file_path} not a good format for .csv with folders.")

        header = ["image", "clinical_score", "predictional_score", "timestamp"]

        # If the file dont exist, create it and add header
        if not os.path.exists(file_path):
            dir_path = os.path.dirname(file_path)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    self.log(f"Created {dir_path}")
            with open(file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
            self.log(f"Created {file_path}")
        # Else check if its a valid header
        else:
            with open(file_path, mode="r", newline="") as file:
                reader = csv.reader(file)
                existing_header = next(reader, None)
                if existing_header != header:
                    raise ValueError(
                        "CSV header does not match expected format!")

        # Write data
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                self.image_path,
                self.get_clinical_pwat(),
                self.get_predicted_pwat(),
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ])
        self.log(f"Edited {file_path}")

    def process(self) -> None:
        """
        Process the image by updating segmentation, masks, and PWAT scores.
        """
        self._update_image()
        self._update_segmentation()
        self._update_masks()
        self._update_wound_masked()
        self._update_peri_wound_mask()
        self._update_peri_wound_masked()
        self._update_predicted_pwat()
        self._update_clinical_pwat()

    def get_image(self) -> ndarray:
        """
        Get the original image as a NumPy array.

        Returns:
            ndarray: The loaded image.
        """
        if self._image is None:
            self._update_image()
        return self._image

    def _update_image(self) -> None:
        """
        Load and update the original image.
        """
        rgb_img = cv2.imread(self.image_path)
        bgr_img = rgb_img[..., ::-1]  # Convert BGR to RGB
        self._image = bgr_img

    def get_segmentation(self) -> ndarray:
        """
        Get the segmentation mask.

        Returns:
            ndarray: The segmentation mask.
        """
        if self._segmentation is None:
            self._update_segmentation()
        return self._segmentation

    def _update_segmentation(self) -> None:
        """
        Perform wound segmentation and update the segmentation mask.
        """
        self._segmentation = wound_segmentation(
            img=self.get_image(), tol=0.95, verbose=self.logging
        )

    def get_wound_mask(self) -> ndarray:
        """
        Get the wound mask.

        Returns:
            ndarray: The wound mask.
        """
        if self._wound_mask is None:
            self._update_masks()
        return self._wound_mask

    def get_body_mask(self) -> ndarray:
        """
        Get the body mask.

        Returns:
            ndarray: The body mask.
        """
        if self._body_mask is None:
            self._update_masks()
        return self._body_mask

    def get_bg_mask(self) -> ndarray:
        """
        Get the background mask.

        Returns:
            ndarray: The background mask.
        """
        if self._bg_mask is None:
            self._update_masks()
        return self._bg_mask

    def _update_masks(self) -> None:
        """
        Update the wound, body, and background masks.
        """
        self._wound_mask, self._body_mask, self._bg_mask = cv2.split(
            self.get_segmentation())

    def get_wound_masked(self) -> ndarray:
        """
        Get the wound mask image.

        Returns:
            ndarray: The wound mask image.
        """
        if self._wound_masked is None:
            self._update_wound_masked()
        return self._wound_masked

    def _update_wound_masked(self) -> None:
        """
        Update the wound masks image.
        """
        img = self.get_image()
        self._wound_masked = cv2.bitwise_and(
            img, img, mask=self.get_wound_mask())

    def get_peri_wound_mask(self) -> ndarray:
        """
        Get the peri-wound mask.

        Returns:
            ndarray: The peri-wound mask.
        """
        if self._peri_wound_mask is None:
            self._update_peri_wound_mask()
        return self._peri_wound_mask

    def _update_peri_wound_mask(self) -> None:
        """
        Update the peri-wound masks.
        """
        wound_mask = self.get_wound_mask()
        pwm = get_perilesion_mask(
            # NOTE: You can play with ksize parameter (tuple[int, int]) but can
            # give wrong predicion and it could probably depend of the file
            # dimension (dynamic ksize)
            ksize=(20, 20),
            mask=wound_mask
        )
        self._peri_wound_mask = cv2.bitwise_and(
            pwm, pwm, mask=imfill(self.get_body_mask() | wound_mask)
        )

    def get_peri_wound_masked(self) -> ndarray:
        """
        Get the peri-wound mask image.

        Returns:
            ndarray: The peri-wound mask image.
        """
        if self._peri_wound_masked is None:
            self._update_peri_wound_masked()
        return self._peri_wound_masked

    def _update_peri_wound_masked(self) -> None:
        """
        Update the peri-wound masks image.
        """
        img = self.get_image()
        self._peri_wound_masked = cv2.bitwise_and(
            img, img, mask=self.get_peri_wound_mask()
        )

    def get_predicted_pwat(self) -> float:
        """
        Get the predicted PWAT.

        Returns:
            float: The predicted PWAT.
        """
        if self._predicted_pwat is None:
            self._update_predicted_pwat()
        return self._predicted_pwat

    def _update_predicted_pwat(self) -> None:
        """
        Update the predicted PWAT.
        """
        self._predicted_pwat = evaluate_PWAT_score(
            # NOTE: You can play with ksize parameter (tuple[int, int]) but can
            # give wrong predicion and it could probably depend of the file
            # dimension (dynamic ksize)
            ksize=(65, 65),
            img=self.get_image(), mask=self.get_segmentation(), verbose=self.logging
        )

    def get_clinical_pwat(self) -> float:
        """
        Get the clinical PWAT.

        Returns:
            float: The clinical PWAT.
        """
        if self._clinical_pwat is None:
            self._update_clinical_pwat()
        return self._clinical_pwat

    def _update_clinical_pwat(self) -> None:
        """
        Update the clinical PWAT.
        """
        # TODO: handle clinical labelisation by setting '_clinical_pwat' as init param
        # For 'wound_image' list that use input folder, add inside a .csv
        # or something like that and handle it
        self._clinical_pwat = 0.0

    def _valid_image_path(self, image_path):
        """
        Check if the image path is a valid folder architecture and file format.

        Args:
            image_path (str): Path to an image file.

        Raises:
            ValueError: If the image path is not a valid folder architecure or file format.
        """
        pattern = r"^(?:[A-Za-z]:\\|/)?(?:[\w\s.-]+[/\\])*[\w\s.-]+\.(?:png|jpe?g)$"
        match = re.match(pattern, image_path, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"File {image_path} not a good format for .png/.jpeg/.jpg with folders.")
