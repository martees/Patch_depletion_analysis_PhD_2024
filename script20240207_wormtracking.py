import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import script20240202_intensityevolutionin1area as intensity_ev
from matplotlib.widgets import Slider
import useful_functions
import script202402_fuseimagesoverlap as fuse_images
import os


def find_image_path_list(image_path):
    """
    Takes the path to a directory, and returns a list of paths to all .tif images contained there, not looking into
    subfolders.
    """
    image_path_list = sorted(glob.glob(image_path + "*.tif", recursive=False))
    # image_path_list = image_path_list[400:]
    if not useful_functions.is_linux():  # On windows the glob output uses \\ as separators so remove that
        image_path_list = [name.replace("\\", '/') for name in image_path_list]
    return image_path_list


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(16, 16))
    is_worm_tracked = [True for _ in range(len(image_list))]
    for i_frame in range(len(image_list)):
        # Background subtraction: average the last 10 frames and subtract this to current frame to get the worm
        if i_frame == 0:  # treating frame 0 differently because there are no previous frame for subtraction
            last_10_frames = image_list[i_frame]
            last_10_diff = image_list[i_frame] - last_10_frames
        else:
            last_10_frames = np.mean([image_list[i] for i in range(max(0, i_frame - 10), i_frame)], axis=0)
            last_10_diff = image_list[i_frame] - last_10_frames

        # Threshold the highest value pixels to get the worm blob
        _, image_thresh = cv2.threshold(last_10_diff, np.quantile(last_10_diff, 0.7), 255, cv2.THRESH_BINARY)
        # Erode once to remove the 1 or 2 pixel wide noise from the image, and dilate 7 times to make sure all worm pixels are included
        image_denoise = cv2.erode(image_thresh, kernel, iterations=1)
        image_denoise = cv2.dilate(image_denoise, kernel, iterations=5)
        # Find blobs: labels = matrix with the same shape as image_denoise and contains label for every point
        num_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(image_denoise.astype(np.uint8))

        # Check if this frame has only one blob (so only the background, no worm detected, and remember that)
        if num_blobs == 1:
            is_worm_tracked[i_frame] = False

        # Exclude first blob (background) and make a centroid of the centroids
        centroids = centroids[1:]
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

    return time_stamps, trajectory_x, trajectory_y, silhouette_list


def generate_tracking(image_path, regenerate_assembled_images):
    """
    This function takes the path to a directory containing a time series of images containing one worm (images are actually
    separated as 4 different tiles for 1 image), and will generate:
    - An "assembled_images" directory, containing:
        - the same images but assembled together using the fuseimagesoverlap script
        - for each assembled image, a .npy array corresponding to the tracking of the worm
    - In the main directory:
        - a list_positions_x.npy array
        - a list_position_y.npy array
        - a list_tracked_frame_numbers.npy array
    """
    # Assemble the images
    if not os.path.isdir(image_path + "assembled_images"):
        # If the assembled_images directory does not exist, create it
        os.mkdir(image_path+"assembled_images")
        # Force the tracking process to start by image assembly
        regenerate_assembled_images = True
    if regenerate_assembled_images:
        # Find images in the main directory
        image_path_list = find_image_path_list(image_path)
        # Assemble images 4 by 4 (see description of fuse_images_overlap())
        for i_image in range(len(image_path_list) // 4):
            # Load the 4 tiles constituting each image into a list
            current_image_paths = [image_path_list[4*i_image + i] for i in [0, 1, 2, 3]]
            current_image_list = [[], [], [], []]
            for i_tile in range(len(current_image_paths)):
                current_image_list[i_tile] = cv2.imread(current_image_paths[i_tile], -1)  # -1 to keep their bit depth
            # Give the list to the assembly function
            image = fuse_images.fuse_images_overlap(current_image_list)
            # Convert it to 8bit and save
            ratio = np.amax(image) / 256
            image = (image / ratio).astype('uint8')
            cv2.imwrite(image_path+"assembled_images/assembled"+str(4*i_image)+".tif", image)

    # Load assembled images
    image_path_list = find_image_path_list(image_path+"assembled_images/")
    list_of_images = [[] for _ in range(len(image_path_list))]
    for i_image in range(len(image_path_list)):
        list_of_images[i_image] = cv2.imread(image_path_list[i_image], -1)

    # Track the worm, and save the results
    time_stamps, list_positions_x, list_positions_y, silhouettes = track_worm(list_of_images)
    np.save(image_path + "list_tracked_frame_numbers.npy", time_stamps)
    np.save(image_path + "list_positions_x.npy", list_positions_x)
    np.save(image_path + "list_positions_y.npy", list_positions_y)
    # Instead of saving a huge silhouette table, save one per image in a separate folder
    if not os.path.isdir(image_path+"/silhouettes"):
        os.mkdir(image_path+"/silhouettes")
    for i in range(len(silhouettes)):
        np.save(image_path + "/silhouettes/silhouettes_"+str(i)+".npy", silhouettes[i])


if useful_functions.is_linux():
    path = "/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"
else:
    # path = 'C:/Users/Asmar/Desktop/These/Patch_depletion/test_pipeline/20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown-02/'
    path = 'E:/Backup/Patch_depletion_dissectoscope/subtest_for_tests/'

generate_tracking(path, regenerate_assembled_images=False)
print("hihi")

# interactive_worm_plot(list_of_images[1:], list_positions_x, list_positions_y, silhouettes)

# fig, axs = plt.subplots(1, 2)
#
# axs[0].imshow(silhouette_density)
# axs[1].imshow(np.mean([image_list[i] for i in range(0, 5)], axis=0))
# axs[1].plot(trajectory_x, trajectory_y, color="yellow")
#
# plt.show()
