class RGB:
    """
    A class representing RGB color values.

    This class provides predefined RGB color constants and a static method
    to create custom RGB color tuples.

    Attributes:
        RED (tuple): RGB value for the color red (255, 0, 0).
        GREEN (tuple): RGB value for the color green (0, 255, 0).
        BLUE (tuple): RGB value for the color blue (0, 0, 255).
        BLACK (tuple): RGB value for the color black (0, 0, 0).
        WHITE (tuple): RGB value for the color white (255, 255, 255).
    """

    # Predefined RGB color constants
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    @staticmethod
    def CUSTOM(r: int, g: int, b: int) -> tuple[int, int, int]:
        """
        Creates a custom RGB color tuple.

        Args:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).

        Returns:
            tuple[int, int, int]: A tuple representing the custom RGB color.

        Raises:
            ValueError: If any of the RGB values are outside the range 0-255.
        """
        # Validate that each RGB value is within the valid range (0-255)
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError("RGB values must be between 0 and 255.")
        return (r, g, b)
