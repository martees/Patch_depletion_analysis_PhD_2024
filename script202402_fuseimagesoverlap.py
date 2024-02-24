# Script created by Aurèle Boussard, commented by Alid Al-Asmar

import os
import numpy as np
import cv2


def fuse_images_overlap(images, overlap=0.3):
    """
    Takes a list of 4 images that correspond to 4 corners of the same thing, in the following order:
    [upper left, upper right, lower right, lower left]
    and with some overlap between them.
    Will return an image which is a stitching of the four images that reconstructs the full image.
    """
    # Define image size and margins
    y_len, x_len = images[0].shape
    y_margin = np.round(y_len * overlap).astype(int)
    x_margin = np.round(x_len * overlap).astype(int)
    # Length of the overlap between the images (x and y)
    y_small_len, x_small_len = y_len - y_margin, x_len - x_margin

    # Slice the images to get only the margins (so 2 values for every pixel in 4 margins, 4 values in the center)
    top_margin_from_left = images[0][:y_small_len, x_small_len:]
    top_margin_from_right = images[1][:y_small_len, :x_margin]

    bot_margin_from_left = images[3][y_margin:, x_small_len:]
    bot_margin_from_right = images[2][y_margin:, :x_margin]

    left_margin_from_top = images[0][y_small_len:, :x_small_len]
    left_margin_from_bot = images[3][:y_margin, :x_small_len]

    right_margin_from_top = images[1][y_small_len:, x_margin:]
    right_margin_from_bot = images[2][:y_margin, x_margin:]

    center_margin_from_tl = images[0][y_small_len:, x_small_len:]
    center_margin_from_tr = images[1][y_small_len:, :x_margin]
    center_margin_from_bl = images[3][:y_margin, x_small_len:]
    center_margin_from_br = images[2][:y_margin, :x_margin]

    # Set the values of the non-overlapping segments of the full image
    final_image = np.zeros((y_small_len * 2 + y_margin, x_small_len * 2 + x_margin), dtype=np.uint16)
    final_image[:y_small_len, :x_small_len] = images[0][:y_small_len, :x_small_len]
    final_image[:y_small_len, -x_small_len:] = images[1][:y_small_len, -x_small_len:]
    final_image[-y_small_len:, -x_small_len:] = images[2][-y_small_len:, -x_small_len:]
    final_image[-y_small_len:, :x_small_len] = images[3][-y_small_len:, :x_small_len]

    # Define median and minimal intensity value of this image to adjust those for the margins
    final_median = np.median(final_image[final_image > 0])
    final_min = np.quantile(final_image[final_image > 0], 0.01)

    # "Fuse" the margins that overlap, with the goal median and minimum values
    top_margin = fuse_margins([top_margin_from_left, top_margin_from_right], final_min, final_median)
    bot_margin = fuse_margins([bot_margin_from_left, bot_margin_from_right], final_min, final_median)
    left_margin = fuse_margins([left_margin_from_top, left_margin_from_bot], final_min, final_median)
    right_margin = fuse_margins([right_margin_from_top, right_margin_from_bot], final_min, final_median)
    center_margin = fuse_margins(
        [center_margin_from_tl, center_margin_from_tr, center_margin_from_bl, center_margin_from_br], final_min,
        final_median)

    final_image[:y_small_len, x_small_len:-x_small_len] = top_margin
    final_image[-y_small_len:, x_small_len:-x_small_len] = bot_margin

    final_image[y_small_len:-y_small_len, :x_small_len] = left_margin
    final_image[y_small_len:-y_small_len, -x_small_len:] = right_margin

    final_image[y_small_len:-y_small_len, x_small_len:-x_small_len] = center_margin

    return final_image


def fuse_margins(margins, final_min, final_median, final_type=np.uint16):
    # Average all the margins
    final_margin = np.mean(margins, axis=0)
    # Align left bound of hist to final min
    final_margin -= np.min(final_margin)
    final_margin += final_min
    # Adjust the median
    while np.median(final_margin) < final_median:
        final_margin += 1
    # Remove values that are too high for final_type
    dtype_max = np.iinfo(final_type).max
    final_margin[final_margin > dtype_max] = dtype_max
    final_margin = np.round(final_margin).astype(final_type)
    return final_margin


if __name__ == "__main__":
    os.chdir(
        "C:/Users/Asmar/Desktop/These/Patch_depletion/test_pipeline/20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown-02/")
    image_list = []
    image_names = ["20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown (1).tif",
                   "20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown (2).tif",
                   "20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown (3).tif",
                   "20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown (4).tif"]
    for i, image_name in enumerate(image_names):
        # i = 0
        # image_name = image_names[i]
        image = cv2.imread(image_name, -1)
        image_list.append(image)
    full_image = fuse_images_overlap(image_list, overlap=0.3)

    from matplotlib import pyplot as plt
    plt.imshow(full_image, vmin=400, vmax=1000)
    plt.show()
