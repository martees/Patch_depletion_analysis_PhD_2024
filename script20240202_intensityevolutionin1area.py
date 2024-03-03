import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
from math import factorial

import useful_functions


def update_vertical_line(h, x, y_min=None, y_max=None):
    """
    https://stackoverflow.com/questions/29331401/updating-pyplot-vlines-in-interactive-plot
    """
    seg_old = h.get_segments()
    if y_min is None:
        y_min = seg_old[0][0, 1]
    if y_max is None:
        y_max = seg_old[0][1, 1]

    seg_new = [np.array([[xx, y_min],
                         [xx, y_max]]) for xx in x]

    return seg_new


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less than `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def intensity_evolution(list_of_images, x_range=None, y_range=None, remove_peaks=False):
    """
    Takes a list of images (that should correspond to a time series), and return the evolution of the mean intensity through time.
    """
    # Set x_range and y_range as full image boundaries if they're set to None
    if x_range is None:
        x_range = [0, list_of_images[0].shape[1]]
    if y_range is None:
        y_range = [0, list_of_images[0].shape[0]]
    #
    x_range = [880, 900]
    y_range = [1020, 1040]

    avg_intensities = np.zeros(len(list_of_images))
    for i in range(len(list_of_images)):
        avg_intensities[i] = np.mean(list_of_images[i][y_range[0]:y_range[1], x_range[0]:x_range[1]])

    # Smooth it
    avg_intensities = savitzky_golay(avg_intensities, 41, 1)  # window size 51, polynomial order 1

    # Remove peaks in the intensity evolution because they correspond to the worm
    if remove_peaks:
        first_derivative = avg_intensities[1:] - avg_intensities[:-1]
        keep_index = [True for i in range(
            len(avg_intensities))]  # Bool list with TRUE if you should keep a point (derivative in this point <= 0), else FALSE
        for i in range(1, len(first_derivative) - 1):
            # If this point is a huge peak
            if first_derivative[i] > 0:
                # If the derivative is positive, remember that you should remove the highest point later
                keep_index[i + 1] = False
                # Look for previous index to keep
                last_true_index = i
                while keep_index[last_true_index] == False and last_true_index > 0:
                    last_true_index = last_true_index - 1
                # Update the 1st derivative according to the last valid point
                first_derivative[i + 1] = avg_intensities[i + 2] - avg_intensities[
                    last_true_index]  # out of bound check?

        # Remove the peak values
        avg_intensities = avg_intensities[keep_index]
        # Create a list of corresponding timepoints (in order to plot avg_intensities as a time series with the proper times)
        timestamps = np.array(range(len(first_derivative) + 1))
        timestamps_to_keep = timestamps[keep_index]

        # Interpolate linearly in the gaps so that the length of the intensity vector does not change
        # (otherwise it's a huge mess later for the plots)
        avg_intensities = np.interp(timestamps, timestamps_to_keep, avg_intensities)
        # np.interp takes the x where there should be data, the x of the current data, and the y, to make linear interpolation in the gaps

    return avg_intensities


def find_drops(intensity_list):
    """
    Takes a list of intensities and returns the indices at which the intensity drops by more than threshold.
    """
    first_derivative = intensity_list[1:] - intensity_list[:-1]
    second_derivative = first_derivative[1:] - first_derivative[:-1]
    potential_drops_2nd_order = np.where(second_derivative < 0)[0]
    potential_drops_1st_order = np.where(first_derivative < 0)[0]
    drops = 0
    return drops


def find_no_worm_sites(silhouettes_path, site_size):
    """
    Takes a path to a time series of worm silhouettes (matrices with tracking label of each pixel, 0 is background),
    and a site size (an integer corresponding to half the side of a square site).
    Will return a list of all the sites of the correct size that have never been too close to the silhouette.
    """
    silhouettes_path_list = useful_functions.find_path_list(silhouettes_path, extension="*.npy")
    silhouettes = 0
    # Sum all silhouette matrices because we want to find sites that have never been visited
    silhouettes_all_times = np.sum(silhouettes, axis=0)


def find_non_visited_sites(silhouettes_path, dilation_radius, site_size):
    """
    Takes a path to a time series of worm silhouettes (matrices with tracking label of each pixel, 0 is background), a
    radius by which to dilate those silhouettes (so a pixel that is this radius away from a silhouette will be
    considered to have been visited), and a site size (an integer corresponding to half the side of a square site).
    Will return a list of all the sites of the correct size that have never been too close to the silhouette.
    """
    return 0


def interactive_intensity_plot(image_path, start, end, x_range=None, y_range=None, analysis_size=10):
    """
    Function that shows a scrollable plot with on top, an area of the image, and on the bottom, the mean intensity
    evolution for this area. When you scroll, it should switch to the next frame, and display current time on curve.
    """
    # Find .tif images in the defined folder
    image_path_list = useful_functions.find_path_list(image_path)

    # Adjust end time so that it does not exceed maximal size
    end = min(end, len(image_path_list))

    # Set x_range and y_range as full image boundaries if they're set to None
    first_img = cv2.imread(image_path_list[0])
    if x_range is None:
        x_range = [0, first_img.shape[1]]
    if y_range is None:
        y_range = [0, first_img.shape[0]]

    global image_list, hist_list, bin_list
    image_list = []
    # Lists to store the histogram of each image
    hist_list = []
    bin_list = []
    for i in range(start, end):
        image = cv2.imread(image_path_list[i])
        # image = cv2.addWeighted(image, 60, np.zeros(image.shape, image.dtype), 0, 100)
        # image = 255*((image - min(0, np.min(image)))/np.max(image))
        image = image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        hist, bins = np.histogram(image, bins=1000)
        hist_list.append(hist)
        bin_list.append(bins)
        image_list.append(image)

    # Define figure and axes
    global left_ax
    global middle_ax
    global right_ax
    global rightmost_ax
    fig, [left_ax, middle_ax, right_ax, rightmost_ax] = plt.subplots(1, 4)
    fig.set_size_inches(25, 5)

    # Define time counter (will be incremented/decremented depending on what user does)
    global curr_time
    curr_time = 0

    # Left axis: histogram of intensity of the image, vertical line = selected pixel intensity in current frame
    left_ax.set_yscale("log")
    left_ax.set_title("Histogram of the current image")
    global hist_plot
    hist_plot = left_ax.plot(bin_list[0][0:-1], hist_list[0])
    global selected_pixel_vertical_line
    selected_pixel_vertical_line = left_ax.axvline(bin_list[0][1], color="darkorange")
    global normalization_pixel_vertical_line
    normalization_pixel_vertical_line = left_ax.axvline(bin_list[0][1], color="grey")

    # Middle axis: current frame of the image. Little orange star = selected pixel
    middle_ax.imshow(image_list[0])
    middle_ax.set_title("Image at current time point")
    # Selected pixel initialization: start with the middle of the image
    image_middle_line = image_list[curr_time].shape[0]
    image_middle_col = image_list[curr_time].shape[1]
    global selected_pixel
    selected_pixel = middle_ax.scatter([image_middle_line], [image_middle_col], color="orange", marker="s",
                                       s=analysis_size * 2, zorder=3, alpha=0.6)
    # Initialize plot object for a pixel to base the normalization on
    global normalization_pixel
    normalization_pixel = middle_ax.scatter([], [], color="grey", marker="s", s=analysis_size * 2, zorder=3, alpha=0.6)

    # Right axis: evolution of average intensity of the selected pixels (orange), of the normalization pixels (grey)
    right_ax.set_title("Evolution of average intensity of orange / grey pixels")
    # Orange: selected pixel intensity evolution (averaged over a 20x20 square around it)
    global selected_pixel_intensities, selected_pixel_intensity_plot
    selected_pixel_intensities = intensity_evolution(image_list, [image_middle_col - analysis_size,
                                                                  image_middle_col + analysis_size],
                                                     [image_middle_line - analysis_size,
                                                      image_middle_line + analysis_size])
    selected_pixel_intensity_plot = right_ax.plot(range(len(selected_pixel_intensities)), selected_pixel_intensities,
                                                  color="darkorange")
    # Grey: normalization pixel intensity evolution
    global normalization_pixel_intensities, normalization_pixel_intensity_plot
    normalization_pixel_intensities = intensity_evolution(image_list, [image_middle_col - analysis_size,
                                                                       image_middle_col + analysis_size],
                                                          [image_middle_line - analysis_size,
                                                           image_middle_line + analysis_size])
    normalization_pixel_intensity_plot = right_ax.plot(range(len(normalization_pixel_intensities)),
                                                       normalization_pixel_intensities, color="grey")
    # Initialize the vertical line that should move depending on current time
    global current_time_line_right
    current_time_line_right = right_ax.axvline(curr_time, color="black")

    global whole_image_intensities
    whole_image_intensities = intensity_evolution(image_list)

    # Rightmost axis: normalized evolution of intensity (selected pixel / normalization pixel)
    rightmost_ax.set_title("Normalized intensity (orange divided by grey)")
    # Green: selected pixel divided by normalization pixel (through time)
    global normalized_intensities, normalized_intensity_plot
    normalized_intensities = selected_pixel_intensities / normalization_pixel_intensities
    normalized_intensity_plot = rightmost_ax.plot(range(len(normalized_intensities)), normalized_intensities,
                                                  color="green")
    # Red: vertical lines indicating where the algorithm detects that there has been an intensity drop
    # global drops_lines
    # drops_lines = rightmost_ax.vlines([], ymin=rightmost_ax.get_ylim()[0], ymax=rightmost_ax.get_ylim()[1], color="red")
    # Initialize the vertical line that should move depending on current time
    global current_time_line_rightmost
    current_time_line_rightmost = rightmost_ax.axvline(curr_time, color="black")

    # Call the update frame once to initialize the plot
    update_frame(image_list, hist_list, bin_list, curr_time)

    # Make the plot scrollable
    def scroll_event(event):
        global curr_time
        if event.button == "up":
            curr_time = curr_time + 1
        elif event.button == "down":
            curr_time = max(0, curr_time - 1)
        else:
            return
        update_frame(image_list, hist_list, bin_list, curr_time)

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
        update_frame(image_list, hist_list, bin_list, curr_time)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    def click_event(event):
        global image_list
        global curr_time
        global right_ax
        global rightmost_ax

        # Plot objects for the star representing those pixels on the image
        global selected_pixel
        global normalization_pixel

        # Plot objects for vertical line corresponding to intensity of selected/norm pixels, and intensity drops
        global selected_pixel_vertical_line
        global normalization_pixel_vertical_line
        # global drops_lines

        # List of intensities already computed for selected, normalization pixels and normalized version
        global selected_pixel_intensities
        global normalization_pixel_intensities
        global normalized_intensities
        global whole_image_intensities

        # Plot objects for the curves representing the intensity evolution on the right plot
        global selected_pixel_intensity_plot
        global normalization_pixel_intensity_plot
        global normalized_intensity_plot

        if event.button is MouseButton.LEFT and event.inaxes == middle_ax:
            # Get the mouse click position, make it integer
            curr_x = int(event.xdata)
            curr_y = int(event.ydata)

            # Change the position of the dot
            selected_pixel.set_offsets([curr_x, curr_y])

            # Update the plot of evolution of intensity of the current selected pixel, and corresponding intensity drops
            selected_pixel_intensities = intensity_evolution(image_list, [curr_x - analysis_size,
                                                                          curr_x + analysis_size],
                                                             [curr_y - analysis_size,
                                                              curr_y + analysis_size])
            selected_pixel_intensity_plot[0].set_data(range(len(selected_pixel_intensities)),
                                                      selected_pixel_intensities)

            # Update the vertical line corresponding to the pixel intensity on the right axis
            selected_pixel_vertical_line.set_xdata([selected_pixel_intensities[curr_time]])

            # Update the normalized plot
            normalized_intensities = selected_pixel_intensities / normalization_pixel_intensities
            # normalized_intensities = selected_pixel_intensities / whole_image_intensities
            normalized_intensity_plot[0].set_data(range(len(normalized_intensities)), normalized_intensities)

            # Update plot limits
            print(np.min([np.min(selected_pixel_intensities), np.min(normalization_pixel_intensities)]))
            print(np.max([np.max(selected_pixel_intensities), np.max(normalization_pixel_intensities)]))
            right_ax.set_ylim(
                int(np.min([np.min(selected_pixel_intensities), np.min(normalization_pixel_intensities)])),
                int(np.max([np.max(selected_pixel_intensities), np.max(normalization_pixel_intensities)])))
            rightmost_ax.set_ylim(np.min(normalized_intensities), np.max(normalized_intensities))

            #
            # # Crappy test for identifying drops
            # curr_intensity_drops = find_drops(selected_pixel_intensities)
            # drops_lines.set_segments(update_vertical_line(drops_lines, curr_intensity_drops, y_min=rightmost_ax.get_ylim()[0],
            #                          y_max=rightmost_ax.get_ylim()[1]))

        elif event.button is MouseButton.RIGHT:
            # Get the mouse click position, make it integer
            curr_x = int(event.xdata)
            curr_y = int(event.ydata)
            print(curr_x)
            print(curr_y)

            # Change the position of the dot on the image
            normalization_pixel.set_offsets([curr_x, curr_y])

            # Update the plot of evolution of intensity of the current normalization pixel
            normalization_pixel_intensities = intensity_evolution(image_list,
                                                                  [curr_x - analysis_size, curr_x + analysis_size],
                                                                  [curr_y - analysis_size, curr_y + analysis_size])
            normalization_pixel_intensity_plot[0].set_data(range(len(normalization_pixel_intensities)),
                                                           normalization_pixel_intensities)

            # Update the vertical line corresponding to the pixel intensity on the left axis
            normalization_pixel_vertical_line.set_xdata([normalization_pixel_intensities[curr_time]])

            # Update the normalized plot
            normalized_intensities = selected_pixel_intensities / normalization_pixel_intensities
            # normalized_intensities = selected_pixel_intensities / whole_image_intensities
            normalized_intensity_plot[0].set_data(range(len(normalized_intensities)), normalized_intensities)

            # Update plot limits
            right_ax.set_ylim(
                int(np.min([np.min(selected_pixel_intensities), np.min(normalization_pixel_intensities)])),
                int(np.max([np.max(selected_pixel_intensities), np.max(normalization_pixel_intensities)])))
            rightmost_ax.set_ylim(np.min(normalized_intensities), np.max(normalized_intensities))

            # Crappy test for identifying drops
            # curr_intensity_drops = find_drops(selected_pixel_intensities)
            # drops_lines.set_segments(
            #     update_vertical_line(drops_lines, curr_intensity_drops, y_min=rightmost_ax.get_ylim()[0],
            #                          y_max=rightmost_ax.get_ylim()[1]))

        curr_fig = plt.gcf()
        curr_fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', click_event)

    print("=== INSTRUCTION GUIDE! :-) ===")
    print(
        "In order to move through the trajectory slowly, use the mouse scroll. If it freezes, just wait, it's loading.")
    print("In order to jump to a faraway timestep, use the slider below the graph.")
    print("In order to look at the intensity evolution of a single pixel, left click on it.")
    print("In order to switch back to full intensity profile, ")

    plt.show()


def find_best_normalization(list_of_images, x_range, y_range, silhouettes):
    """
    Takes a list of images, an x and y range on that image (in the format of a list: x_range could be [100, 200]),
    and will determine the best area of pixels to normalize the selected range on.
    First method I'm going to implement: look for the closest area of the same size that has never been close to the
    silhouettes.
    """


def update_frame(list_of_images, histogram_values, bin_edges, index):
    """
    Function that is called every time the time index has to change.
    """
    global middle_ax
    global current_time_line_right
    global current_time_line_rightmost
    global left_ax
    global hist_plot

    # Shift the vertical line on the intensity plots
    current_time_line_right.set_xdata([index])
    current_time_line_rightmost.set_xdata([index])

    # Plot the new histogram
    hist_plot[0].set_data([bin_edges[index][0:-1], histogram_values[index]])

    # middle_ax.cla()
    middle_ax.imshow(list_of_images[index], vmax=60)

    #curr_fig = plt.gcf()
    #curr_fig.canvas.draw()


if __name__ == "__main__":
    # Paths to find images
    path = "/media/admin/Expansion/Backup/Patch_depletion_dissectoscope/subtest_for_tests/"

    # Number of lines and columns to the microscope tiling (for when you take only a quarter of the drop on each img for ex)
    n_lines = 2
    n_columns = 2

    # Time steps at which to start / end analysis
    start_time = 0
    end_time = 3000

    # Limits of the restricted area to scan on the image
    # x_limits = [600, 900]
    # y_limits = [500, 1750]

    x_limits = None
    y_limits = None

    interactive_intensity_plot(path, start_time, end_time, x_limits, y_limits, analysis_size=10)

# intensities1 = []
# intensities2 = []
# intensities3 = []
# intensities4 = []
# for i in range(len(image_list)):
#     if i % (n_lines * n_columns) == 0:
#         intensities1.append(np.mean(image_list[i][0:500, 0:500]))
# if i % (n_lines * n_columns) == 1:
#     intensities2.append(np.mean(image_list[i][0:500, -500:-1]))
# if i % (n_lines * n_columns) == 2:
#     intensities3.append(np.mean(image_list[i][-500:-1, -500:-1]))
# if i % (n_lines * n_columns) == 3:
#     intensities4.append(np.mean(image_list[i][-500:-1, 0:500]))

# avg_intensities = np.zeros(min(len(intensities1), len(intensities4)))
# for i in range(avg_intensities):
#     avg_intensities[i] = np.mean([intensities1[i], intensities2[i], intensities3[i], intensities4[i]])

# avg_intensities1 = np.zeros(len(intensities1))
# avg_intensities2 = np.zeros(len(intensities2))
# avg_intensities3 = np.zeros(len(intensities3))
# avg_intensities4 = np.zeros(len(intensities4))
# avg_intensities_whole_plate = np.zeros(len(intensities1))

# for i in range(min(len(avg_intensities1), len(avg_intensities4))):
# avg_intensities1[i] = np.mean(image_list[4*i][0:500, 0:500])
# avg_intensities2[i] = np.mean(image_list[4*i + 1][0:500, -500:-1])
# avg_intensities3[i] = np.mean(image_list[4*i + 2][-500:-1, -500:-1])
# avg_intensities4[i] = np.mean(image_list[4*i + 3][-500:-1, 0:500])
# avg_intensities_whole_plate = np.mean([intensities1[i], intensities2[i], intensities3[i], intensities4[i]])

# fig, axs = plt.subplots(1, 2)

# plt.plot(range(len(intensities1)), intensities1, label="quarter1")
# plt.plot(range(len(intensities2)), intensities2, label="quarter2")
# plt.plot(range(len(intensities3)), intensities3, label="quarter3")
# plt.plot(range(len(intensities4)), intensities4, label="quarter4")
# plt.title("Normalized intensity evolution of each quarter")
# plt.legend()

# axs[1].imshow(image_list[-1])
# axs[1].set_title("Patch aspect at the end")
