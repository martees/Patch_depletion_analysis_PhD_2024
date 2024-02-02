# This is a script that takes our microscope images of patches, and assembles them to recreate a full image of each
# time step (our microscope takes multiple images, following a tiling through the arena)
# For now I gave up on this because we are focusing on analyzing images one by one.

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

# Paths to find images and empty field image, and path to output assembled images
path = "/home/admin/Desktop/Patch_depletion/Microscope/20240125_1patch1worm_0.209od_5x_4s_20w4000ms_8h_plate/TIFF/"
image_path_list = sorted(glob.glob(path + "*.tif", recursive=False))
assembled_images_path = "/home/admin/Desktop/Patch_depletion/Microscope/20240125_1patch1worm_0.209od_5x_4s_20w4000ms_8h_plate/stitched/"
empty_field_image_path = "/home/admin/Desktop/Patch_depletion/Microscope/20240128_emptyfieldimages/280124_x5_emptyfield_GFP20_4s/280124_x5_emptyfield_GFP20_4s_C00_ORG.tif"

# Number of lines and columns to the microscope tiling (for when you take only a quarter of the drop on each img for ex)
n_lines = 2
n_columns = 2


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def histogram_equalize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)


# Load all the images into a list
image_list = []
for i_path in range(5):
    current_image = cv2.imread(image_path_list[i_path], -1)
    image_list.append(current_image)


for i_image in range(1):
    current_image_list = image_list[n_lines * n_columns * i_image: (i_image + 1) * n_lines * n_columns]
    current_edited_image_list = []
    for j_image in range(len(current_image_list)):
        current_image = current_image_list[j_image]

        brightness = 100
        contrast = 10
        current_image = cv2.addWeighted(current_image, contrast, np.zeros(current_image.shape, current_image.dtype), 0,
                                        brightness)

        #current_image = histogram_equalize(current_image)
        #current_image = cv2.normalize(current_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        current_edited_image_list.append(current_image)

    assembled_image = []
    for i_line in range(n_lines):

        if i_line % 2 == 0:
            current_line = current_edited_image_list[i_line * n_columns]
            column_range = range(i_line * n_columns + 1, n_columns * (i_line + 1))
        else:
            current_line = current_edited_image_list[(i_line + 1) * n_columns - 1]
            column_range = range((i_line + 1) * n_columns - 2, i_line * n_columns - 1, -1)

        for i_column in column_range:
            current_line = np.concatenate((current_line, current_edited_image_list[i_column]), axis=1)  # stack on the right

        if i_line == 0:
            assembled_image = current_line
        else:
            assembled_image = np.concatenate((assembled_image, current_line), axis=0)  # stack on the bottom

    #assembled_image = current_images_list[3]

    #_, assembled_image = cv2.threshold(assembled_image, 6, 255, cv2.THRESH_BINARY)

    #assembled_image = cv2.cvtColor(assembled_image, cv2.COLOR_BGR2GRAY)

    #ret, assembled_image = cv2.threshold(assembled_image, 5, 255, cv2.THRESH_BINARY)
    #assembled_image = cv2.adaptiveThreshold(assembled_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                        cv2.THRESH_BINARY, 21, 2)
    #assembled_image = cv2.adaptiveThreshold(assembled_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                        cv2.THRESH_BINARY, 11, 1)

    #assembled_image = cv2.equalizeHist(assembled_image)

    # create a CLAHE object (Arguments are optional).
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    #assembled_image = clahe.apply(assembled_image)

    # Adjust the brightness and contrast
    # Adjusts the brightness by adding 10 to each pixel value
    brightness = 100
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 0
    assembled_image = cv2.addWeighted(assembled_image, contrast, np.zeros(assembled_image.shape, assembled_image.dtype), 0, brightness)

    #assembled_image = cv2.normalize(assembled_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(assembled_images_path + str(i_image) + ".tif", assembled_image)
