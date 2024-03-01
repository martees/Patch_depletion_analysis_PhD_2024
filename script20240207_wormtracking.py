import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import script20240202_intensityevolutionin1area as intensity_ev

import useful_functions
import script202402_fuseimagesoverlap as fuse_images
import os
import time


def find_image_path_list(image_path):
    """
    Takes the path to a directory, and returns a list of paths to all .tif images contained there, not looking into
    subfolders.
    """
    image_path_list = sorted(glob.glob(image_path + "*.tif", recursive=False))
    # image_path_list = image_path_list[400:]
    if not useful_functions.is_linux():  # On windows the glob output uses \\ as separators so remove that
        image_path_list = [name.replace("\\", '/') for name in image_path_list]
    return np.array(image_path_list)


def track_worm(image_list):
    """
    Function that takes a list of uint8 images (as returned by cv2.imread), and tracks the worm on them, returning
    a matrix with the blobs identified at each time step, and a centroid.
    """
    # List of x and y coordinates for the worm centroid at each time point
    trajectory_x = np.zeros(len(image_list))
    trajectory_y = np.zeros(len(image_list))
    # List of pixel-wise tracking output for each time point
    silhouette_list = []
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(8, 8))
    # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(24, 24))
    is_worm_tracked = [True for _ in range(len(image_list))]
    last_20_frames = 0
    for i_frame in range(len(image_list)):
        if i_frame % 10 == 0:
            print("Tracking frame " + str(i_frame) + " / " + str(len(image_list)))
        # Background subtraction: average the last 10 frames and subtract this to current frame to get the worm

        if i_frame == 0:  # treating frame 0 differently because there are no previous frame for subtraction
            last_20_frames = image_list[i_frame]
            last_20_diff = image_list[i_frame] - last_20_frames
        # else:
        #     last_20_frames = np.mean([image_list[j] for j in range(max(0, i_frame - 20), i_frame)], axis=0)
        #     last_20_diff = image_list[i_frame] - last_20_frames
        elif i_frame < 21:  # treating frames that are between 1 and 19 differently for 20-frame-deep bg subtraction
            last_20_frames = np.mean([image_list[j] for j in range(max(0, i_frame - 20), i_frame)], axis=0)
            last_20_diff = image_list[i_frame] - last_20_frames
        else:
            last_20_frames = last_20_frames + 1 / 20 * (image_list[i_frame] - image_list[i_frame - 21])
            last_20_diff = image_list[i_frame] - last_20_frames

        # Threshold the highest value pixels to get the worm blob
        _, image_thresh = cv2.threshold(last_20_diff, np.quantile(last_20_diff, 0.7), 255, cv2.THRESH_BINARY)
        # Erode once to remove the 1 or 2 pixel wide noise from the image, and dilate multiple times to make it more
        # probable that the worm is only one blob
        image_denoise = cv2.erode(image_thresh, erosion_kernel, iterations=3)
        # image_denoise = cv2.dilate(image_denoise, dilation_kernel, iterations=1)
        # Find blobs: labels = matrix with the same shape as image_denoise and contains label for every point
        num_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(image_denoise.astype(np.uint8))

        # Check if this frame has only one blob (so only the background, no worm detected, and remember that for later)
        if num_blobs == 1:
            is_worm_tracked[i_frame] = False
        else:
            # Exclude first blob (background)
            centroids = centroids[1:]
            # If centroids are too far apart, just take the closest one to the previous one
            if np.max(centroids[:, 0]) - np.min(centroids[:, 0]) > 600 or np.max(centroids[:, 1]) - np.min(
                    centroids[:, 1]) > 600:
                distance_to_previous = np.sqrt((centroids[0, 0] - trajectory_x[i_frame - 1]) ** 2 + (
                            centroids[0, 1] - trajectory_y[i_frame - 1]) ** 2)
                min_distance = distance_to_previous
                best_centroid = 0
                for i_centroid in range(len(centroids)):
                    distance_to_previous = np.sqrt((centroids[i_centroid, 0] - trajectory_x[i_frame - 1]) ** 2 + (
                            centroids[i_centroid, 1] - trajectory_y[i_frame - 1]) ** 2)
                    if distance_to_previous < min_distance:
                        min_distance = distance_to_previous
                        best_centroid = i_centroid
                    if distance_to_previous < 600:
                        break
                trajectory_x[i_frame] = centroids[best_centroid, 0]
                trajectory_y[i_frame] = centroids[best_centroid, 1]

            # If centroids are all close to each other, take the centroid of the centroids
            else:
                centroid_of_centroids_x = np.mean(centroids[:, 0])
                centroid_of_centroids_y = np.mean(centroids[:, 1])
                trajectory_x[i_frame] = centroid_of_centroids_x
                trajectory_y[i_frame] = centroid_of_centroids_y

        silhouette_list.append(labels)

    # Remove from the trajectories the points where the worm was not tracked
    trajectory_x = trajectory_x[is_worm_tracked]
    trajectory_y = trajectory_y[is_worm_tracked]
    # Create a list of time stamps that correspond to the frame numbers of the tracked frames
    time_stamps = np.array(range(len(image_list)))[is_worm_tracked]

    return time_stamps.astype(int), trajectory_x, trajectory_y, silhouette_list


# def find_better_silhouette(image, x, y):
#     """
#     Function that takes an image and the coordinates of a blob found using background subtraction.
#     """


def generate_tracking(image_path, regenerate_assembled_images=False, track_assembled=False):
    """
    This function takes the path to a directory containing a time series of images containing one worm (images are actually
    separated as 4 different tiles for 1 image), and will generate:
    - An "assembled_images" directory, containing:
        - the same images but assembled together using the fuseimagesoverlap script
        - for each assembled image, a .npy array corresponding to the tracking of the worm
    - In the main directory:
        - a list_positions_x.npy array
        - a list_position_y.npy array
        - a list_tracked_frame_numbers.npy array/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/subtest_for_tests/
    """
    # Assemble the images
    if not os.path.isdir(image_path + "assembled_images"):
        # If the assembled_images directory does not exist, create it
        os.mkdir(image_path + "assembled_images")
        # Force the tracking process to start by image assembly
        regenerate_assembled_images = True
    if regenerate_assembled_images:
        # Find images in the main directory
        image_path_list = find_image_path_list(image_path)
        # Assemble images 4 by 4 (see description of fuse_images_overlap())
        for i_image in range(len(image_path_list) // 4):
            if i_image % 10 == 0:
                print("Assembling images "+str(i_image)+" / "+str(len(image_path_list)))
            # Load the 4 tiles constituting each image into a list
            current_image_paths = [image_path_list[4 * i_image + i] for i in [0, 1, 2, 3]]
            current_image_list = [[], [], [], []]
            for i_tile in range(len(current_image_paths)):
                current_image_list[i_tile] = cv2.imread(current_image_paths[i_tile], -1)  # -1 to keep their bit depth
            # Give the list to the assembly function
            image = fuse_images.fuse_images_overlap_2x2(current_image_list)
            # Convert it to 8bit and save
            ratio = np.amax(image) / 256
            image = (image / ratio).astype('uint8')
            # Generate an image number which always has 4 digits (for proper sorting later)
            image_number = str(i_image)
            if i_image < 10:
                image_number = "0" + image_number
            if i_image < 100:
                image_number = "0" + image_number
            if i_image < 1000:
                image_number = "0" + image_number
            cv2.imwrite(image_path + "assembled_images/assembled" + image_number + ".tif", image)

    if track_assembled:
        # Load images
        image_path_list = find_image_path_list(image_path + "assembled_images/")
        list_of_images = [[] for _ in range(len(image_path_list))]
        for i_image in range(len(image_path_list)):
            if i_image % 25 == 0:
                print("Loading image " + str(i_image) + " / " + str(len(image_path_list)))
            list_of_images[i_image] = cv2.imread(image_path_list[i_image], -1)
        print("Finished loading images")
        time_stamps, list_positions_x, list_positions_y, silhouettes = track_worm(list_of_images)
        print("Finished tracking")

    else:
        # Load images
        image_path_list = find_image_path_list(image_path)
        list_of_images = [[] for _ in range(len(image_path_list))]
        for i_image in range(len(image_path_list)):
            list_of_images[i_image] = cv2.imread(image_path_list[i_image], -1)
        # Separate them into the four quadrants to which they correspond
        upper_left_images = [list_of_images[4 * j] for j in range(len(list_of_images) // 4)]
        upper_right_images = [list_of_images[4 * j + 1] for j in range(len(list_of_images) // 4)]
        lower_right_images = [list_of_images[4 * j + 2] for j in range(len(list_of_images) // 4)]
        lower_left_images = [list_of_images[4 * j + 3] for j in range(len(list_of_images) // 4)]

        # Track the worm in every quadrant independently
        times_upper_left, x_upper_left, y_upper_left, silhouettes_upper_left = track_worm(upper_left_images)
        times_upper_right, x_upper_right, y_upper_right, silhouettes_upper_right = track_worm(upper_right_images)
        times_lower_right, x_lower_right, y_lower_right, silhouettes_lower_right = track_worm(lower_right_images)
        times_lower_left, x_lower_left, y_lower_left, silhouettes_lower_left = track_worm(lower_left_images)

        # Convert the coordinates to match the full image
        x_upper_left, y_upper_left = useful_functions.shift_xy_coordinates_2x2(list_of_images[0], "upper left", 0.3,
                                                                               x_upper_left, y_upper_left)
        x_upper_right, y_upper_right = useful_functions.shift_xy_coordinates_2x2(list_of_images[0], "upper right", 0.3,
                                                                                 x_upper_right, y_upper_right)
        x_lower_left, y_lower_left = useful_functions.shift_xy_coordinates_2x2(list_of_images[0], "lower left", 0.3,
                                                                               x_lower_left, y_lower_left)
        x_lower_right, y_lower_right = useful_functions.shift_xy_coordinates_2x2(list_of_images[0], "lower right", 0.3,
                                                                                 x_lower_right, y_lower_right)

        time_stamps = np.concatenate([times_upper_left, times_upper_right, times_lower_right, times_lower_left])
        list_positions_x = np.concatenate([x_upper_left, x_upper_right, x_lower_right, x_lower_left])
        list_positions_y = np.concatenate([y_upper_left, y_upper_right, y_lower_right, y_lower_left])

        silhouettes = [[] for _ in range(len(image_path_list) // 4)]
        for i_time in range(len(image_path_list) // 4):
            list_silhouettes = [silhouettes_upper_left[i_time], silhouettes_upper_right[i_time],
                                silhouettes_lower_right[i_time], silhouettes_lower_left[i_time]]
            silhouettes[i_time] = fuse_images.fuse_images_overlap_2x2(list_silhouettes, 0.3)

    # Saving data tables
    np.save(image_path + "list_tracked_frame_numbers.npy", time_stamps)
    np.save(image_path + "list_positions_x.npy", list_positions_x)
    np.save(image_path + "list_positions_y.npy", list_positions_y)

    # Instead of saving a huge silhouette table, save one per image in a separate folder
    if not os.path.isdir(image_path+"/silhouettes"):
        os.mkdir(image_path+"/silhouettes")
    for i_time in range(len(silhouettes)):
        np.save(image_path + "/silhouettes/silhouettes_"+str(i_time)+".npy", silhouettes[i_time])

    return time_stamps, list_positions_x, list_positions_y, silhouettes


def generate_no_worm_images(image_path, x, y, t):
    """
    Takes a path containing images
    """


def intensity_evolution_1_area_where_worm_was(list_of_x, list_of_y, tracked_time_points, when, small_radius, big_radius):
    """
    Function that will plot the intensity evolution in an area where the worm went at time point defined by the "when"
    parameter (so the goal is to look at the dynamic of the intensity before / during / after the worm was somewhere).
    Uses a small and a big radius to define squares around the worm. Inside is divided by outside.
    """
    # Compute the temporal dynamics of one area before / after worm arrives
    worm_passage = tracked_time_points[when]  # take some time step where the worm is tracked
    area_center_x = int(list_of_x[tracked_time_points == worm_passage][0])  # center area on worm position
    area_center_y = int(list_of_y[tracked_time_points == worm_passage][0])

    worm_distance = []
    avg_intensity_close = []
    avg_intensity_further = []
    avg_intensity_full_img = []
    for time in range(len(assembled_images_path)):  # for every time step, look at intensity in the area
        if time % 10 == 0:
            print("Looking at time step "+str(time)+" / "+str(len(assembled_images_path)))
        current_image = cv2.imread(assembled_images_path[time // 4], -1)
        avg_intensity_full_img.append(np.mean(current_image))

        if time in tracked_time_points:
            current_worm_x = int(list_of_x[tracked_time_points == time][0])
            current_worm_y = int(list_of_y[tracked_time_points == time][0])
            # Compute current distance to worm
            worm_distance.append(np.sqrt((current_worm_x - area_center_x) ** 2 + (current_worm_y - area_center_y) ** 2))

        # Current state of the small area
        current_small_area = current_image[area_center_x - small_radius:area_center_x + small_radius,
                             area_center_y - small_radius:area_center_y + small_radius]
        avg_intensity_close.append(np.mean(current_small_area))

        # Area around the "close to worm" area (section between the two radii)
        # We divide it in 4 segments to exclude the area close to the worm
        further_from_worm_left = current_image[area_center_x - big_radius: area_center_x - small_radius,
                                 area_center_y - big_radius: area_center_y + big_radius]
        further_from_worm_right = current_image[area_center_x + small_radius: area_center_x + big_radius,
                                  area_center_y - big_radius: area_center_y + big_radius]
        further_from_worm_top = current_image[area_center_x - small_radius: area_center_x + small_radius,
                                area_center_y + small_radius: area_center_y + big_radius]
        further_from_worm_bottom = current_image[area_center_x - small_radius: area_center_x + small_radius,
                                   area_center_y - big_radius: area_center_y - small_radius]
        # Make it all a line because we only care about the intensities
        current_further_area = np.concatenate((np.ravel(further_from_worm_left), np.ravel(further_from_worm_right),
                                               np.ravel(further_from_worm_top), np.ravel(further_from_worm_bottom)))
        avg_intensity_further.append(np.mean(current_further_area))

    fig, [left_ax, right_ax, rightmost_ax] = plt.subplots(1, 3)
    fig.set_size_inches(18, 4)
    plt.suptitle("Intensity evolution in the area where the worm crosses at time=" + str(when))

    left_ax.imshow(cv2.imread(assembled_images_path[worm_passage], -1), vmax=70)
    left_ax.plot([area_center_x - small_radius, area_center_x + small_radius, area_center_x + small_radius,
                  area_center_x - small_radius, area_center_x - small_radius],
                 [area_center_y + small_radius, area_center_y + small_radius, area_center_y - small_radius,
                  area_center_y - small_radius, area_center_y + small_radius], color="orange",
                 label="small = " + str(small_radius))
    left_ax.plot([area_center_x - big_radius, area_center_x + big_radius, area_center_x + big_radius,
                  area_center_x - big_radius, area_center_x - big_radius],
                 [area_center_y + big_radius, area_center_y + big_radius, area_center_y - big_radius,
                  area_center_y - big_radius, area_center_y + big_radius], color="black",
                 label="big = " + str(big_radius))
    left_ax.set_title("Image of when the worm is in the area")
    left_ax.legend()

    # right_ax.plot(range(len(avg_intensity_close)), avg_intensity_close, color="orange")
    # right_ax.plot(range(len(avg_intensity_further)), avg_intensity_further, color="black")
    right_ax.plot(range(len(avg_intensity_close)), np.array(avg_intensity_close) / np.array(avg_intensity_full_img),
                  color="blue", label="orange/full intensity")
    right_ax.plot(range(len(avg_intensity_close)), np.array(avg_intensity_close) / np.array(avg_intensity_further),
                  color="black", label="orange/black intensity")
    right_ax.axvline(worm_passage, color="red", label="worm passage")
    right_ax.set_title("Average intensity evolution in the two squares")
    right_ax.legend()

    rightmost_ax.plot(tracked_time_points, worm_distance, color="green")
    rightmost_ax.axvline(worm_passage, color="red", label="worm passage")
    rightmost_ax.set_title("Evolution of distance from worm to current area")

    plt.show()


def intensity_as_a_function_of_worm_distance(parent_path, area_x_range, area_y_range, x_worm, y_worm, t_worm):
    """
    Function that takes a path containing a time series of images, x and y ranges defining a rectangular area on it,
    and tracked time points of the worm (x, y, and t)
    Will plot intensity as a function of distance to the worm.
    """
    path_to_images = find_image_path_list(parent_path)
    area_center = [np.mean(area_x_range), np.mean(area_y_range)]
    area_radius = np.max([abs(area_x_range[0] - area_x_range[1]), abs(area_y_range[0] - area_y_range[1])])

    avg_intensity = []
    distance_to_worm = []
    previous_avg_intensity = 0
    first_loop = True
    for i_time in range(len(path_to_images)):
        if i_time in t_worm:  # if the worm is tracked in the current time step
            time_index = np.where(t_worm == i_time)[0][0]
            curr_img = cv2.imread(path_to_images[time_index], -1)
            curr_avg_intensity = np.mean(curr_img[area_x_range[0]: area_x_range[1], area_y_range[0]: area_y_range[1]])
            if first_loop:
                first_loop = False
                previous_avg_intensity = curr_avg_intensity
            avg_intensity.append(curr_avg_intensity - previous_avg_intensity)
            previous_avg_intensity = curr_avg_intensity
            distance_to_worm.append(np.sqrt((area_center[0] - x_worm[time_index])**2 + (area_center[1] - y_worm[time_index])**2))

    fig, [left_ax, right_ax] = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)

    # An ugly one-liner to get the image corresponding to the smallest distance from worm to area
    min_distance_index = t_worm[np.argmin(distance_to_worm)]
    left_ax.imshow(cv2.imread(path_to_images[min_distance_index], -1), vmax=70)
    left_ax.plot([area_x_range[0], area_x_range[1], area_x_range[1], area_x_range[0], area_x_range[0]],
                 [area_y_range[1], area_y_range[1], area_y_range[0], area_y_range[0], area_y_range[1]],
                 color="orange", label="x="+str(area_x_range)+", y="+str(area_y_range))
    left_ax.set_title("Video image with worm as close as possible (t="+str(np.argmin(distance_to_worm))+")")
    left_ax.legend()

    avg_intensity = np.array(avg_intensity)
    distance_to_worm = np.array(distance_to_worm)
    right_ax.plot(distance_to_worm, avg_intensity, color="lightgray", zorder=0)
    right_ax.scatter(distance_to_worm, avg_intensity, c=t_worm[t_worm < len(path_to_images)], cmap="jet")
    right_ax.scatter(distance_to_worm[distance_to_worm < area_radius], avg_intensity[distance_to_worm < area_radius],
                     marker="*", c=t_worm[t_worm < len(path_to_images)][distance_to_worm < area_radius],
                     cmap="jet", label="Worm centroid inside area")
    right_ax.set_xlabel("Distance to the worm")
    right_ax.set_ylabel("Average intensity of the area")
    right_ax.set_title("Intensity as a function of worm-to-area distance")

    plt.show()


tic = time.time()
if useful_functions.is_linux():
    # path = "/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/subtest_for_tests/"
    path = '/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/'
else:
    # path = "C:/Users/apeadmin/Desktop/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"
    # path = 'E:/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/'
    path = 'E:/Backup/Patch_depletion_dissectoscope/subtest_for_tests/'

regenerate_tracking = True
if regenerate_tracking:
    t, x, y, sil = generate_tracking(path, regenerate_assembled_images=False, track_assembled=True)
    print("Tracking over and saved, it took ", time.time() - tic, "seconds to run!")
else:
    os.chdir(path)
    t, x, y = np.load("list_tracked_frame_numbers.npy"), np.load("list_positions_x.npy"), np.load("list_positions_y.npy")
    print("Finished loading data tables.")

images_path = find_image_path_list(path)
assembled_images_path = find_image_path_list(path + "assembled_images/")

# intensity_evolution_1_area_where_worm_was(x, y, t, 34, 400, 500)
# intensity_as_a_function_of_worm_distance(path + "assembled_images/", [1000, 1500], [2500, 3000], x, y, t)
useful_functions.interactive_worm_plot(assembled_images_path, t, x, y)

# whole_image_intensity = np.array(whole_image_intensity)
# no_worm_intensity = np.array(no_worm_intensity)
# close_to_worm_intensity = np.array(close_to_worm_intensity)
#
# fig, [left_ax, right_ax] = plt.subplots(1, 2)
# left_ax.plot(t, whole_image_intensity, label="whole image intensity")
# left_ax.plot(t, no_worm_intensity, label="no worm intensity")
# left_ax.plot(t, close_to_worm_intensity, label="close to worm intensity")
# left_ax.legend()
# right_ax.plot(t, close_to_worm_intensity/whole_image_intensity, label="worm/whole")
# right_ax.plot(t, close_to_worm_intensity/no_worm_intensity, label="worm/noworm")
# right_ax.legend()
# plt.suptitle("Radius around worm: "+str(close_radius)+"px. Radius for no_worm:"+str(further_radius)+"px")
# plt.show()


# images_list = []
# assembled_images = []
# for i in range(len(images_path)):
#     images_list.append(cv2.imread(images_path[i], -1))
#     if i % 4 == 0:
#         assembled_images.append(cv2.imread(assembled_images_path[i//4], -1))
#
# useful_functions.interactive_worm_plot(assembled_images_path, t, x, y, sil)

# os.chdir("/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/")
# assembled_images_test = ["./assembled_images/assembled"+str(i)+".tif" for i in [272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324]]
# assembled_images_test = ["20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown ("+str(i)+").tif" for i in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]]
# images_path = find_image_path_list(path)
# images_path = images_path[272:304]
# assembled_images_path = find_image_path_list(path+"assembled_images/")
# images_list = []
# assembled_images = []
# for i in range(len(images_path)):
#     images_list.append(cv2.imread(images_path[i], -1))
#     assembled_images.append(cv2.imread(assembled_images_path[272/4+i], -1))
# t, x, y, sil = track_worm(images_list)
# # t, x, y, sil = np.load("list_positions_x.npy"),

# interactive_worm_plot(list_of_images[1:], list_positions_x, list_positions_y, silhouettes)

# fig, axs = plt.subplots(1, 2)
#
# axs[0].imshow(silhouette_density)
# axs[1].imshow(np.mean([image_list[i] for i in range(0, 5)], axis=0))
# axs[1].plot(trajectory_x, trajectory_y, color="yellow")
#
# plt.show()


# time_stamps = []
# list_positions_x = []
# list_positions_y = []
# silhouettes = []
# # Handle duplicate tracking
# for i_time in range(len(list_of_images) // 4):  # for each time step
#     list_of_x_for_this_time_step = []
#     list_of_y_for_this_time_step = []
#     if i_time in times_upper_left:
#         list_of_x_for_this_time_step.append(x_upper_left)
#         list_of_y_for_this_time_step.append(y_upper_left)
#     if i_time in times_upper_right:
#         list_of_x_for_this_time_step.append(x_upper_right)
#         list_of_y_for_this_time_step.append(y_upper_right)
#     if i_time in times_lower_right:
#         list_of_x_for_this_time_step.append(x_lower_right)
#         list_of_y_for_this_time_step.append(y_lower_right)
#     if i_time in times_lower_left:
#         list_of_x_for_this_time_step.append(x_lower_left)
#         list_of_y_for_this_time_step.append(y_lower_left)
#     # For every point in the tracking (should also work when there's just one)
#     for i_track in range(len(list_of_x_for_this_time_step)):
#         time_stamps.append(i_time)
#         list_positions_x.append(np.mean(list_of_x_for_this_time_step))
#         list_positions_y.append(np.mean(lis))
