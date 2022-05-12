import argparse
import cv2
import numpy
import os

import brightness_transformation

def parse_arguments(tested_function):
    """
    Parse user arguments for brightness_transformation functions
    """
    parser = argparse.ArgumentParser(
        description='parse arguments for brightness_transformation functions')

    parser.add_argument('--image_filename',
                    required=True,
                    help="""Path to the image directory.""")

    parser.add_argument('--write_image_dir',
                    required=True,
                    help="""Path to the write the transformed image.""")


    parser.add_argument('--gamma',
                    type=float,
                    help="""Gammar value for power law transformation.""")

    return parser.parse_args()

def test_adjust_image_intensity():
    """
    Test the adjust image intensity function.
    """
    # Parse user arguments for training phase.
    args = parse_arguments(tested_function='adjust_image_intensity')

    # Read image.
    image = cv2.imread(args.image_filename, cv2.IMREAD_GRAYSCALE)

    # Transform the input image with power law transformation.
    transformed_image = brightness_transformation.adjust_image_intensity(
        image, 0, 255, 255, 0, args.gamma
    )

    # Write the transformed image to the path
    cv2.imwrite(args.write_image_dir + 'image_power_law_transform.png', transformed_image)

def test_log_transformation():
    """
    Test the log transformation function.
    """
    # Parse user arguments for training phase.
    args = parse_arguments(tested_function='log_transformation')

    # Read image.
    image = cv2.imread(args.image_filename, cv2.IMREAD_GRAYSCALE).astype(numpy.float32)

    # Apply fourier transformation to the image.
    f = numpy.fft.fft2(image)

    # Shift the fourier image.
    f_shift = numpy.fft.fftshift(f)

    # Compute the magnitude spectrum of the fourier image.
    magnitude_spectrum = numpy.abs(f_shift)

    # Transform the input image and magitude spectrum with logarithmic function.
    transformed_fft = brightness_transformation.log_transformation(magnitude_spectrum)
    transformed_image = brightness_transformation.log_transformation(image)

    # Write the relevant images to the path
    os.makedirs(args.write_image_dir, exist_ok =True)
    cv2.imwrite(args.write_image_dir + 'image.png', image)
    cv2.imwrite(args.write_image_dir + 'magnitude_spectrum.png', magnitude_spectrum)
    cv2.imwrite(args.write_image_dir + 'fft_log_transformation.png', transformed_fft)
    cv2.imwrite(args.write_image_dir + 'image_log_transformation.png', transformed_image)

if __name__ == '__main__':
    test_adjust_image_intensity()
    test_log_transformation()
