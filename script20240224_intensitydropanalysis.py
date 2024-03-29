import script20240202_intensityevolutionin1area as intensity
import useful_functions
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import copy


def intensity_vs_wormed_frames(list_of_images, full_img_intensity_evolution, x, y, start_time, end_time):
    """
    Will return a list of average intensities for the (x,y) pixel, corresponding to the average intensity between
    every worm passage to this pixel (represented by nans). If there are multiple consecutive nans, will put nans
    in the list of averages, so that there are as many elements in the output as time steps where the worm was on
    this pixel.
    """
    pixel_intensity_list = np.zeros(len(list_of_images))
    control_pixel_intensity_list = np.zeros(len(list_of_images))
    for i_time in range(len(list_of_images)):
        curr_img = list_of_images[i_time]
        pixel_intensity_list[i_time] = curr_img[y, x]
        control_pixel_intensity_list[i_time] = curr_img[y - 277, x]

    # Find the indices where the worm is there
    nan_indices = np.where(np.isnan(pixel_intensity_list))[0]
    nan_indices = np.append(nan_indices, len(pixel_intensity_list))

    # Add the nans in the control one
    control_pixel_intensity_list[nan_indices[:-1]] = np.nan

    # Average between each passage of the worm
    average_each_passage = np.zeros(len(nan_indices))
    control_average_each_passage = np.zeros(len(nan_indices))
    prev_nan_index = 0
    for i_nan in range(len(nan_indices)):
        curr_nan_index = nan_indices[i_nan]
        # If there's no time point between the last passage and the current passage beginning, put a nan
        if curr_nan_index - prev_nan_index == 0:
            average_each_passage[i_nan] = np.nan
            control_average_each_passage[i_nan] = np.nan
        else:
            average_each_passage[i_nan] = np.nanmean(pixel_intensity_list[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
            average_each_passage[i_nan] /= np.nanmean(full_img_intensity_evolution[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
            # Do the same thing to the control pixel
            control_average_each_passage[i_nan] = np.nanmean(control_pixel_intensity_list[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
            control_average_each_passage[i_nan] /= np.nanmean(full_img_intensity_evolution[max(prev_nan_index, curr_nan_index - 10): curr_nan_index])
        prev_nan_index = curr_nan_index

    # Make it be the difference with the previous period
    # diff_each_passage has nan values whenever there are consecutive wormed time steps in average_each_passage, so that
    # the i-th term of diff_each_passage has the difference in intensity after i time steps with the worm
    diff_each_passage = copy.copy(average_each_passage)
    control_diff_each_passage = copy.copy(control_average_each_passage)
    # Find indices where the intensity is defined (non nan values of average_each_passage)
    non_nan_indices = np.where(np.logical_not(np.isnan(diff_each_passage)))[0]
    # Start at 1 because difference is defined for the second element onwards (we add 0 as the first after the loop)
    for i_not_nan in range(1, len(non_nan_indices)):
        curr_nan = non_nan_indices[i_not_nan]
        prev_nan = non_nan_indices[i_not_nan - 1]
        diff_each_passage[curr_nan] -= average_each_passage[prev_nan]
        control_diff_each_passage[curr_nan] -= control_average_each_passage[prev_nan]
    diff_each_passage[0] = 0
    control_diff_each_passage[0] = 0

    if max(np.nancumsum(control_diff_each_passage)) > 2:
        print("hihi")

    diff_each_passage = np.nancumsum(diff_each_passage)
    control_diff_each_passage = np.nancumsum(control_diff_each_passage)

    return average_each_passage, diff_each_passage, control_diff_each_passage


def plot_intensity_worm_passages(path, start_time, end_time, timestamps, x_range=None, y_range=None):
    print("Loading no_worm images...")
    image_path_list = useful_functions.find_path_list(path + "assembled_images/")
    end_time = np.min((end_time, len(image_path_list)))  # end time cannot be more than number of frames
    image_list = [[] for _ in range(start_time, end_time)]
    for t in range(start_time, end_time):
        image_list[t - start_time] = np.load(image_path_list[t][:-len(".tif")] + "_nansilhouette.npy")

    print("Computing average intensity in every image...")
    full_img_intensity_list = np.zeros(len(image_list))
    for i in range(len(image_list)):
        full_img_intensity_list[i] = np.nanmean(image_list[i][y_range[0]:y_range[1], x_range[0]:x_range[1]])

    # We create a list of empty lists, with one element per frame with a tracked worm
    # Sublist i will contain the values of differences in intensity after i-th worm visit
    list_of_avg_intensity_diff = [[] for _ in range(len(timestamps) + 1)]
    list_of_avg_intensity_diff_ctrl = [[] for _ in range(len(timestamps) + 1)]
    for x in range(x_range[0], x_range[1]):
        if x % 100 == 0:
            print("Computing intensity evolution for column ", x, " / ", x_range[1])
        for y in range(y_range[0], y_range[1]):
            intensity_series_this_pixel, intensity_diff_this_pixel, control_diff_this_pixel = intensity_vs_wormed_frames(image_list, full_img_intensity_list, x, y, start_time, end_time)
            for i in range(len(intensity_diff_this_pixel)):
                if not np.isnan(intensity_diff_this_pixel[i]):
                    list_of_avg_intensity_diff[i].append(intensity_diff_this_pixel[i])
                    list_of_avg_intensity_diff_ctrl[i].append(control_diff_this_pixel[i])

    list_of_avg_intensity_diff = [list_of_avg_intensity_diff[i] for i in range(len(list_of_avg_intensity_diff)) if list_of_avg_intensity_diff[i]]
    list_of_avg_intensity_diff_ctrl = [list_of_avg_intensity_diff_ctrl[i] for i in range(len(list_of_avg_intensity_diff_ctrl)) if list_of_avg_intensity_diff_ctrl[i]]

    fig, [left_ax, right_ax] = plt.subplots(1, 2)

    left_ax.boxplot(list_of_avg_intensity_diff_ctrl, showmeans=True, meanline=True)
    left_ax.axhline(0, color="grey", linestyle="--")
    left_ax.set_title("Control")
    left_ax.set_ylabel("Cumulated sum of normalized difference in intensity")
    left_ax.set_xlabel("Number of worm passages")

    right_ax.boxplot(list_of_avg_intensity_diff, showmeans=True, meanline=True)
    right_ax.axhline(0, color="grey", linestyle="--")
    right_ax.set_title("When worm was there")
    right_ax.set_ylabel("Cumulated sum of normalized difference in intensity")
    right_ax.set_xlabel("Number of worm passages")
    plt.show()


