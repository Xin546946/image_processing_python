import cv2
import numpy

def adjust_image_intensity(image, in_min, in_max, out_min, out_max, gamma = 1):
    """
    Adjust image intensity values by the power law transformation
    The image indensity values, which are lower than in_min, output out_min.
    The image indensity values, which are larger than in_out, output out_max.
    The image indensity values between [in_min, in_max], output power law
        transformation with the power gamma.
    Args:
        image (opencv2): The input image, read by opencv.
        in_min (float): The lower bound of the input image.
        in_max (float): The upper bound of the input image.
        out_min (float): The lower bound of the output image.
        out_max (float): The upper bound of the output image.
        gamma (float, optional): The power/exponent. Defaults to 1.

    Returns:
        image (opencv2): The transformed image.
    """
    return (((image - in_min) / (in_max - in_min)) ** gamma) * (out_max - out_min) + out_min

def log_transformation(image):
    """
    The logarithmic transformation of an image.
    To be able to visualize the image, we multiply a constant c to make sure the
    maximal value of the transformed image is 255.

    Args:
        image (opencv): The input image with the range of (0,255).

    Returns:
        log_image(opencv): The transformed image.
    """
    # Compute a constant c.
    c = 255.0 / numpy.log(1 + numpy.max(image))

    # Transform the image.
    log_image = c * numpy.log(1 + image)

    return log_image