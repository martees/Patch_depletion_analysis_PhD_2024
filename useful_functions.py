import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import cv2
import glob

def is_linux():  # returns true if you're using linux, otherwise false
    try:
        test = os.uname()
        if test[0] == "Linux":
            return True
    except AttributeError:
        return False


def find_path_list(image_path, extension="*.tif"):
    """
    Takes the path to a directory, and returns a list of paths to all .tif images contained there, not looking into
    subfolders.
    """
    image_path_list = sorted(glob.glob(image_path + extension, recursive=False))
    # image_path_list = image_path_list[400:]
    if not is_linux():  # On windows the glob output uses \\ as separators so remove that
        image_path_list = [name.replace("\\", '/') for name in image_path_list]
    return np.array(image_path_list)



def interactive_worm_plot(images_paths, timestamps, centroids_x, centroids_y):
    """
    Function that shows a scrollable plot of the worm and the corresponding tracking.
    Inputs: a list of paths to a time series of .tif images, and arrays with time series of centroids and silhouettes for the tracked worm.
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
    curr_frame = cv2.imread(images_paths[0], -1)
    left_ax.imshow(curr_frame)
    left_ax.scatter(centroids_x[curr_time], centroids_y[curr_time], color="red", s=4)
    left_ax.set_title("Image at current time point")

    # Right axis: tracking of current frame if there is one
    right_ax.set_title("Current tracking")
    global img_shape
    img_shape = curr_frame.shape
    if os.path.isfile(images_paths[0][:-len(".tif")]+"_silhouette.npy"):
        right_ax.imshow(np.load(images_paths[0][:-len(".tif")]+"_silhouette.npy"))
        right_ax.scatter(centroids_x[curr_time], centroids_y[curr_time], color="red")
    else:
        right_ax.imshow(np.zeros(img_shape))

    # Call the update frame once to initialize the plot
    update_frame(images_paths, timestamps, centroids_x, centroids_y, curr_time)

    # Make the plot scrollable
    def scroll_event(event):
        global curr_time
        if event.button == "up":
            curr_time = curr_time + 1
        elif event.button == "down":
            curr_time = max(0, curr_time - 1)
        else:
            return
        update_frame(images_paths, timestamps, centroids_x, centroids_y, curr_time)

    fig.canvas.mpl_connect('scroll_event', scroll_event)

    # Create a slider
    global bottom_ax
    fig = plt.gcf()
    bottom_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(bottom_ax, 'Time', 0, len(images_paths), valinit=0, color="green")
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)

    # Slider update function
    def slider_update(val):  # val argument gives "unused" warning in PyCharm but is necessary
        global curr_time
        curr_time = int(frame_slider.val)
        update_frame(images_paths, timestamps, centroids_x, centroids_y, curr_time)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    print("=== INSTRUCTION GUIDE! :-) ===")
    print(
        "In order to move through the trajectory slowly, use the mouse scroll. If it freezes, just wait, it's loading.")
    print("In order to jump to a faraway timestep, use the slider below the graph.")
    print("In order to look at the intensity evolution of a single pixel, left click on it.")
    print("In order to switch back to full intensity profile, ")

    plt.show()


def update_frame(images_paths, list_timestamps, list_centroids_x, list_centroids_y, index):
    """
    Function that is called every time the time index has to change.
    """
    global right_ax
    global left_ax
    global img_shape
    # Update current video frame on left axis: divided by 4 because full images are named after
    # index of the first image (for now)
    left_ax.cla()
    left_ax.imshow(cv2.imread(images_paths[index], -1), vmax=60)
    left_ax.scatter(list_centroids_x[list_timestamps == index], list_centroids_y[list_timestamps == index], color="red", s=4)
    # Update current tracking on right axis
    right_ax.cla()
    if os.path.isfile(images_paths[index][:-len(".tif")]+"_silhouette.npy"):
        right_ax.imshow(np.load(images_paths[index][:-len(".tif")] + "_silhouette.npy"))
        right_ax.scatter(list_centroids_x[list_timestamps == index], list_centroids_y[list_timestamps == index],
                         color="red")
        right_ax.set_title("Current frame: " + str(index) + " (tracked)")
    else:
        right_ax.imshow(np.zeros(img_shape))
        right_ax.set_title("Current frame: " + str(index) + " (not tracked)")

    #curr_fig = plt.gcf()
    #curr_fig.canvas.draw()


def four_digit_string(number):
    """ Takes 3 and returns "0003", 14 and returns "0014", etc. """
    number_string = str(number)
    if number < 10:
        number_string = "0" + number_string
    if number < 100:
        number_string = "0" + number_string
    if number < 1000:
        number_string = "0" + number_string
    return number_string


def shift_xy_coordinates_2x2(image, position, overlap, x_coord, y_coord):
    """
    Returns shifted coordinates based on position of the image on a bigger image composed of 2x2 tiles.
    :param image: one image that's a quarter of the bigger image (to get the shape)
    :param position: position of the image on the bigger image, eg "upper left" for upper left quadrant
    :param overlap: proportion of overlap between the quadrants (eg 0.3 for 30%)
    :param x_coord: a list of x coordinates that needs to be shifted to match the bigger image
    :param y_coord: same for y
    """
    # Define image size and margins
    y_len, x_len = image.shape
    y_margin = np.round(y_len * overlap).astype(int)
    x_margin = np.round(x_len * overlap).astype(int)
    # Length of the non-overlapping sections of the images (x and y)
    y_small_len, x_small_len = y_len - y_margin, x_len - x_margin

    # For the upper left quadrant, nothing needs be done
    if position == "upper left":
        x_shift = 0
        y_shift = 0
    # For the upper right quadrant, x needs to be increased (by width of non overlapping section of the 2 upper images)
    if position == "upper right":
        x_shift = x_small_len
        y_shift = 0
    # For the two lower quadrants, the same as the upper ones but with a y shift too
    if position == "lower left":
        x_shift = 0
        y_shift = y_small_len
    if position == "lower right":
        x_shift = x_small_len
        y_shift = y_small_len

    return x_coord + x_shift, y_coord + y_shift


