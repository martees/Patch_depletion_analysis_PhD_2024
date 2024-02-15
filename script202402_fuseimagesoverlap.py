# Script created by Aurèle Boussard

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def fuse_images_overlap(images, overlap=0.3):
    y_len, x_len = images[0].shape
    y_margin = np.round(y_len * overlap).astype(int)
    x_margin = np.round(x_len * overlap).astype(int)
    y_small_len, x_small_len = y_len - y_margin, x_len - x_margin

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


    final_image = np.zeros((y_small_len * 2 + y_margin, x_small_len * 2 + x_margin), dtype=np.uint16)
    final_image[:y_small_len, :x_small_len] = images[0][:y_small_len, :x_small_len]
    final_image[:y_small_len, -x_small_len:] = images[1][:y_small_len, -x_small_len:]
    final_image[-y_small_len:, -x_small_len:] = images[2][-y_small_len:, -x_small_len:]
    final_image[-y_small_len:, :x_small_len] = images[3][-y_small_len:, :x_small_len]

    final_median = np.median(final_image[final_image > 0])
    final_min = np.quantile(final_image[final_image > 0], 0.01)

    top_margin = fuse_margins([top_margin_from_left, top_margin_from_right], final_min, final_median)
    bot_margin = fuse_margins([bot_margin_from_left, bot_margin_from_right], final_min, final_median)
    left_margin = fuse_margins([left_margin_from_top, left_margin_from_bot], final_min, final_median)
    right_margin = fuse_margins([right_margin_from_top, right_margin_from_bot], final_min, final_median)
    center_margin = fuse_margins([center_margin_from_tl, center_margin_from_tr, center_margin_from_bl, center_margin_from_br], final_min, final_median)


    final_image[:y_small_len, x_small_len:-x_small_len] = top_margin
    final_image[-y_small_len:, x_small_len:-x_small_len] = bot_margin

    final_image[y_small_len:-y_small_len, :x_small_len] = left_margin
    final_image[y_small_len:-y_small_len, -x_small_len:] = right_margin

    final_image[y_small_len:-y_small_len, x_small_len:-x_small_len] = center_margin

    # from matplotlib import pyplot as plt
    # plt.imshow(final_image, vmin=400, vmax=1000)
    return final_image


def fuse_margins(margins, final_min, final_median, final_type=np.uint16):
    final_margin = np.zeros(margins[0].shape, dtype=np.float64)
    for margin in margins:
        final_margin += margin
    final_margin /= len(margins)
    final_margin -= np.min(final_margin)
    final_margin += final_min
    while np.median(final_margin) < final_median:
        final_margin += 1
    final_margin[final_margin > 65535] = 65535
    final_margin = np.round(final_margin).astype(final_type)
    return final_margin



if __name__ == "__main__":
    os.chdir("D:/Directory/Data/fuse_images_overlap")
    images = []
    image_names = ["20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown_S0000(TR1)_T000000_C00_M0000_ORG.tif", "20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown_S0000(TR1)_T000000_C00_M0001_ORG.tif", "20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown_S0000(TR1)_T000000_C00_M0002_ORG.tif", "20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown_S0000(TR1)_T000000_C00_M0003_ORG.tif"]
    for i, image_name in enumerate(image_names):
        # i = 0
        # image_name = image_names[i]
        image = cv2.imread(image_name, -1)
        images.append(image)
    final_image = fuse_images_overlap(images, overlap=0.3)