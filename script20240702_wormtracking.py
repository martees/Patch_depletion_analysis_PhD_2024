import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import script20240202_intensityevolutionin1area as intensity_ev
from matplotlib.widgets import Slider
import useful_functions
import script202402_fuseimagesoverlap as fuse_images


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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(10, 10))
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

    return time_stamps, trajectory_x, trajectory_y, silhouette_list, is_worm_tracked


def interactive_worm_plot(image_list, centroids_x, centroids_y, silhouettes):
    """
    Function that shows a scrollable plot of the worm and the corresponding tracking.
    Inputs: a time series of images, a time series of centroids and silhouettes for the tracked worm.
    Output: a plot with on the left, the current video image, and on the right, the corresponding
    worm silhouette and centroid.
    """

    # Define figure and axes
    global left_ax
    global right_ax
    fig, [left_ax, right_ax] = plt.subplots(1, 2)
    # fig.set_size_inches(25, 5)

    # Define time counter (will be incremented/decremented depending on what user does)
    global curr_time
    curr_time = 0

    # Left axis: current frame of the image
    left_ax.imshow(image_list[0])
    left_ax.set_title("Image at current time point")

    # Right axis: tracking of current frame
    right_ax.imshow(silhouettes[0])
    right_ax.scatter(centroids_x[curr_time], centroids_y[curr_time], color="red")
    right_ax.set_title("Current tracking")

    # Call the update frame once to initialize the plot
    update_frame(image_list, silhouettes, centroids_x, centroids_y, curr_time)

    # Make the plot scrollable
    def scroll_event(event):
        global curr_time
        if event.button == "up":
            curr_time = curr_time + 1
        elif event.button == "down":
            curr_time = max(0, curr_time - 1)
        else:
            return
        update_frame(image_list, silhouettes, centroids_x, centroids_y, curr_time)

    fig.canvas.mpl_connect('scroll_event', scroll_event)

    # Create a slider
    global bottom_ax
    fig = plt.gcf()
    bottom_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(bottom_ax, 'Time', 0, len(image_list), valinit=0, color="green")
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)

    # Slider update function
    def slider_update(val):  # val argument gives "unused" warning in PyCharm but is necessary
        global curr_time
        curr_time = int(frame_slider.val)
        update_frame(image_list, silhouettes, centroids_x, centroids_y, curr_time)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    print("=== INSTRUCTION GUIDE! :-) ===")
    print(
        "In order to move through the trajectory slowly, use the mouse scroll. If it freezes, just wait, it's loading.")
    print("In order to jump to a faraway timestep, use the slider below the graph.")
    print("In order to look at the intensity evolution of a single pixel, left click on it.")
    print("In order to switch back to full intensity profile, ")

    plt.show()


def update_frame(image_list, list_of_silhouettes, list_centroids_x, list_centroids_y, index):
    """
    Function that is called every time the time index has to change.
    """
    global right_ax
    global left_ax
    # Update current video frame on left axis
    left_ax.imshow(image_list[index])
    # Update current tracking on right axis
    right_ax.cla()
    right_ax.imshow(list_of_silhouettes[index])
    right_ax.scatter(list_centroids_x[index], list_centroids_y[index], color="red")
    right_ax.set_title("Current frame: " + str(index))
    curr_fig = plt.gcf()
    curr_fig.canvas.draw()


def generate_tracking(image_path):
    # Find the path of all the images
    image_path_list = sorted(glob.glob(image_path + "*.tif", recursive=False))
    image_path_list = image_path_list[400:]
    list_of_images = []
    if not useful_functions.is_linux():  # On windows the glob output uses \\ as separators so remove that
        image_path_list = [name.replace("\\", '/') for name in image_path_list]

    # Load the images into a list of arrays, assembling them 4 by 4 (see description of fuse_images_overlap())
    for i_image in range(len(image_path_list) // 4):
        current_image_paths = [image_path_list[i_image + i] for i in [0, 1, 2, 3]]
        current_image_list = [[], [], [], []]
        for i_tile in range(len(current_image_paths)):
            current_image_list[i_tile] = cv2.imread(current_image_paths[i_tile], -1)
        image = fuse_images.fuse_images_overlap(current_image_list)

        # if i_image % 4 == 2:
        #    image = cv2.imread(image_path_list[i_image], -1)  # -1 is to load them without converting them to 8bit
        #    # Convert image to 8 bit depth
        #    ratio = np.amax(image) / 256
        #    image = (image / ratio).astype('uint8')
        list_of_images.append(image)

    time_stamps, list_positions_x, list_positions_y, silhouettes, is_worm_tracked = track_worm(list_of_images)

    np.save(image_path + "list_tracked_frame_numbers.npy", time_stamps)
    np.save(image_path + "list_positions_x.npy", list_positions_x)
    np.save(image_path + "list_positions_y.npy", list_positions_y)
    np.save(image_path + "silhouettes.npy", silhouettes)
    np.save(image_path + "is_worm_tracked.npy", is_worm_tracked)


if useful_functions.is_linux():
    path = "/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"
else:
    path = 'C:/Users/Asmar/Desktop/These/Patch_depletion/test_pipeline/20243101_OD0.2oldbact10gfp4s_Lsomethingworm_dishupsidedown-02/'

generate_tracking(path)
print("hihi")

# interactive_worm_plot(list_of_images[1:], list_positions_x, list_positions_y, silhouettes)

# fig, axs = plt.subplots(1, 2)
#
# axs[0].imshow(silhouette_density)
# axs[1].imshow(np.mean([image_list[i] for i in range(0, 5)], axis=0))
# axs[1].plot(trajectory_x, trajectory_y, color="yellow")
#
# plt.show()
