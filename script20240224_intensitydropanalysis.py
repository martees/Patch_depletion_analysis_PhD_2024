import script20240202_intensityevolutionin1area as intensity
import useful_functions
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import copy


def intensity_vs_wormed_frames(list_of_images, full_img_intensity_evolution, x, y):
    """
    Will return a list of average intensities for the (x,y) pixel, corresponding to the average intensity between
    every worm passage to this pixel (represented by nans). If there are multiple consecutive nans, will put nans
    in the list of averages, so that there are as many elements in the output as time steps where the worm was on
    this pixel.
    """
    pixel_intensity_list = np.zeros(len(list_of_images))
    for i_time in range(len(list_of_images)):
        curr_img = list_of_images[i_time]
        pixel_intensity_list[i_time] = curr_img[y, x]

    # Find the indices where the worm is there
    nan_indices = np.where(np.isnan(pixel_intensity_list))[0]
    nan_indices = np.append(nan_indices, len(pixel_intensity_list))

    # Average between each passage of the worm
    average_each_passage = np.zeros(len(nan_indices))
    prev_nan_index = 0
    for i_nan in range(len(nan_indices)):
        curr_nan_index = nan_indices[i_nan]
        # If there's no time point between the last passage and the current passage beginning, put a nan
        if curr_nan_index - prev_nan_index == 0:
            average_each_passage[i_nan] = np.nan
        else:
            average_each_passage[i_nan] = np.nanmean(pixel_intensity_list[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
            # average_each_passage[i_nan] /= np.nanmean(full_img_intensity_evolution[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
        prev_nan_index = curr_nan_index

    # Make it be the difference with the previous period
    diff_each_passage = copy.copy(average_each_passage)
    non_nan_indices = np.where(np.logical_not(np.isnan(diff_each_passage)))[0]
    for i_not_nan in range(1, len(non_nan_indices)):
        curr_nan = non_nan_indices[i_not_nan]
        prev_nan = non_nan_indices[i_not_nan - 1]
        diff_each_passage[curr_nan] -= average_each_passage[prev_nan]
    diff_each_passage[0] = 0

    if np.max(diff_each_passage) > 1:
        print("hehe")

    return average_each_passage, diff_each_passage


def plot_intensity_worm_passages(path, timestamps):
    image_list = intensity.no_worm_images(path + "assembled_images/", timestamps)

    full_img_intensity_list = np.zeros(len(image_list))
    for i in range(len(full_img_intensity_list)):
        full_img_intensity_list[i] = np.mean(image_list[i][1600:2100][2000:2500])

    list_of_avg_intensities = [[] for _ in range(len(timestamps))]
    for x in range(2000, 2500):
        if x % 100 == 0:
            print("Computing intensity evolution for column ", x, " / ", 3004)
        for y in range(1600, 2100):
            intensity_series_this_pixel, intensity_diff_this_pixel = intensity_vs_wormed_frames(image_list, full_img_intensity_list, x, y)
            for i in range(len(intensity_diff_this_pixel)):
                if not np.isnan(intensity_diff_this_pixel[i]):
                    list_of_avg_intensities[i].append(intensity_diff_this_pixel[i])

    list_of_avg_intensities = [list_of_avg_intensities[i] for i in range(len(list_of_avg_intensities)) if list_of_avg_intensities[i]]

    plt.boxplot(list_of_avg_intensities)
    plt.ylabel("Average diff between when worm left and when worm arrived")
    plt.xlabel("Number of worm passages")
    plt.show()


