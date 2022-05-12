import cv2
import numpy
import argparse
import matplotlib.pyplot as plt

def parse_arguments():
    """
    Parse user arguments for brightness_transformation functions
    """
    parser = argparse.ArgumentParser(
        description='parse arguments for test histogram function')

    parser.add_argument('--image_filename',
                        required=True,
                        help="""Path to the image directory.""")

    parser.add_argument('--write_filename',
                    help="""Path to the write the transformed image.""")


    return parser.parse_args()

class Histogram:
    """
    Construct a histogram object, the input data should be in the range of
    (min_val, max_val) with the number of bins num_bins.
    """

    def __init__(self, num_bins, min_val, max_val):
        # Compute the width of the bin.
        self.bin_width = ((max_val - min_val + 1) / num_bins)
        # Initialize a histogram with 0.0.
        self.histogram = numpy.asarray([0.0] * num_bins)

    def get_bin_id(self, value):
        """
        Compute a correspond bin's id for a given value.
        """
        return int(numpy.floor(value / self.bin_width))

    def add_data(self, value):
        """
        Add a new value to the histogramm. The histogram is updated with the
        new value.
        """
        # Get the bin's id
        bin = self.get_bin_id(value)

        # Update histogram at the corresponding bin.
        self.histogram[bin] += 1

    def make_histogram(self, image):
        """
        Make a histogram for a given image.
        """

        # Get height and width of the image.
        height, width = image.shape

        # Iterate the image and update the histogram
        for r in range(0, height):
            for c in range(0, width):
                # Add pixel to histogram.
                self.add_data(image[r,c])

        # Normalize the histogram to the range of (0,1)
        self.normalize()

        # Print the information of the histogram.
        self.print_info()

    def get_bin_height(self, bin_id):
        """
        Return how many bins are there in the histogramm at bin's id.
        """
        return self.histogram[bin_id]

    def get_histogram(self):
        """
        Get the histogram object.
        """
        return self.histogram

    def normalize(self):
        """
        Normalize the histogram of the range (0,1)
        """

        self.histogram /= numpy.sum(self.histogram)

    def write_histogram_as_plot(self, filename):
        """
        Write histogram as a plot on drive.
        """
        # Define the x axes.
        bar_x = range(self.histogram.shape[0])

        # Create the histogram
        plt.bar(bar_x, self.histogram)

        # Show every x axes.
        plt.xticks(bar_x)

        # Generate relative information on the histogram plot.
        plt.title('Histogram of the Image')
        plt.xlabel('Bins')
        plt.ylabel('Frequencies')

        # Save the histogram with the given filename.
        plt.savefig(filename)

    def print_info(self):
        """
        The information of the histogram.
        """
        print("-------Histogram--------")
        print("The number of bins", self.histogram.shape[0])
        print("Histogram:", self.histogram)
        print("------------------------")



def test_histogram():
    # Parse user arguments.
    args = parse_arguments()

    #Read image.
    image = cv2.imread(args.image_filename, cv2.IMREAD_GRAYSCALE)

    # Declare a histogram with required parameters.
    hist = Histogram(10, 0, 255)

    # Test several functions.
    hist.make_histogram(image)
    hist.write_histogram_as_plot(args.write_filename)


if __name__ == '__main__':
    test_histogram()





