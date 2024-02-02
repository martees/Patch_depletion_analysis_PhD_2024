import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob


# Paths to find images
path = "/media/admin/LaCie/Data/MicroMonitWorm/depletion_pilots/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"

# Number of lines and columns to the microscope tiling (for when you take only a quarter of the drop on each img for ex)
n_lines = 2
n_columns = 2

# Time steps at which to start / end analysis
start_time = 1237
end_time = 3000

# Limits of the restricted area to scan on the image
x_limits = [600, 900]
y_limits = [500, 1750]


def update_vertical_line(h, x, y_min=None, y_max=None):
    seg_old = h.get_segments()
    if y_min is None:
        y_min = seg_old[0][0, 1]
    if y_max is None:
        y_max = seg_old[0][1, 1]

    seg_new = [np.array([[xx, y_min],
                         [xx, y_max]]) for xx in x]

    return seg_new


def intensity_evolution(image_list, x_range, y_range):
    """
    Takes a list of images (that should correspond to a time series), and then look only at an area of this image,
    defined by x_range and y_range. Will return the evolution of the mean intensity over this area through time.
    """
    avg_intensities = []
    for i in range(len(image_list)):
        avg_intensities.append(np.mean(image_list[i][x_range[0]:x_range[1], y_range[0]:y_range[1]]))
    # /np.mean(image_list[i][0:500, 0:500])
    return avg_intensities


def interactive_intensity_plot(folder, start, end, quarter):
    """
    Function that shows a scrollable plot with on top, an area of the image, and on the bottom, the mean intensity
    evolution for this area. When you scroll, it should switch to the next frame, and display current time on curve.
    """
    # Find .tif images in the defined folder
    image_path_list = sorted(glob.glob(folder + "*.tif", recursive=False))

    # Adjust end time so that it does not exceed maximal size
    end = min(end, len(image_path_list))

    # Define figure and axes
    global top_ax
    fig = plt.gcf()
    fig.set_size_inches(15, 15)
    top_ax = fig.gca()

    # Select the images in image_path_list that correspond to the quarter defined in arguments
    # (using parity and number of lines/columns in the microscope tiling)
    image_list = []
    for i in range(start, end):
        if i % (n_lines * n_columns) == quarter:
            image = cv2.imread(image_path_list[i],
                               -1)  # -1 parameter is to not ruin the .tif upon import!!! (bit depth is 16)
            # image = cv2.addWeighted(image, 60, np.zeros(image.shape, image.dtype), 0, 100)
            image_list.append(image)

    # Plot area of the image for first frame
    top_ax.imshow(image_list[0])

    # Define axes limits, to modify them when the user presses arrows
    global xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = plt.axis()

    # Define frame index counter (will be incremented/decremented depending on what user does)
    global curr_index
    curr_index = 0

    # Initialize the plot objects
    global intensity_plot
    global current_time_vline
    intensity_plot = bottom_ax.plot([], [], color=plt.cm.Greys(np.linspace(0, 1, 256))[60], marker="o", markersize=5, linewidth=0)


    center_of_mass_plot = top_ax.scatter([], [], zorder=3, color="orange")
    center_to_center_line = top_ax.plot([], [], color="white")
    # Call the update frame once to initialize the plot
    update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, speed_list)

    # Make the plot scrollable
    def scroll_event(event):
        global curr_index
        if event.button == "up":
            curr_index = curr_index + 1
        elif event.button == "down":
            curr_index = max(0, curr_index - 1)
        else:
            return
        update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, speed_list)

    fig.canvas.mpl_connect('scroll_event', scroll_event)

    # Create a slider
    global bottom_ax
    fig = plt.gcf()
    bottom_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(bottom_ax, 'Index', 0, 30000, valinit=first_frame)
    plt.subplots_adjust(left=0.25, bottom=.2, right=None, top=.9, wspace=.2, hspace=.2)
    # Slider update function

    def slider_update(val):  # val argument gives "unused" warning in PyCharm but is necessary
        global curr_index
        global bottom_ax
        curr_index = int(frame_slider.val)
        update_frame(folder, curr_index, pixels, centers_of_mass, patch_list, speed_list, zoom_out=True)

    # Slider function update call
    frame_slider.on_changed(slider_update)

    # If we're interested in trajectory smoothing, plot the trajectory, and smooth it on right click
    if is_smoothing:
        global traj_plot
        global smooth_plot
        # Plot the raw trajectory
        traj_plot = top_ax.plot(centers_of_mass["x"], centers_of_mass["y"], color="white", marker="x")
        smooth_plot = top_ax.plot(smoothed_traj["x"], smoothed_traj["y"], color="black", marker="o")

        def click_event(event):
            global centers_of_mass
            global traj_plot
            global smooth_plot
            if event.button is MouseButton.LEFT:
                traj_plot.set_data([centers_of_mass["x"]], [centers_of_mass["y"]], color="white", marker="x")
            elif event.button is MouseButton.RIGHT:
                traj_plot.set_data([smoothed_traj["x"]], [smoothed_traj["y"]], color="black", marker="o")

        fig.canvas.mpl_connect('button_press_event', click_event)

    print("=== INSTRUCTION GUIDE! :-) ===")
    print("In order to move through the trajectory slowly, use the mouse scroll. If it freezes, just wait, it's loading.")
    print("In order to jump to a faraway timestep, use the slider below the graph.")

    plt.show()


def update_frame(folder, index, pixels, centers_of_mass, patch_list, speed_list, zoom_out=False):
    """
    Function that is called every time the frame number has to be updated to a new index.
    @param folder: path of the current video
    @param index: new frame number
    @param pixels: dataframe containing one line per frame, and in each frame a list of x,y coordinates for worm silhouette
    @param centers_of_mass: center of mass of the worm
    @param patch_list: list of patches where the worm is at each frame (for checking purposes, displayed in title)
    @param speed_list: list of speeds at which the worm is going at each frame (for checking purposes, displayed as x legend)
    @param zoom_out: if True, xlim and ylim will be set to image limits, otherwise narrow around worm
    """
    global worm_plot
    global center_of_mass_plot
    global center_to_center_line
    global top_ax
    global xmin, xmax, ymin, ymax

    worm_plot[0].set_data([pixels[index][0]], [pixels[index][1]])
    center_of_mass_plot.set_offsets([centers_of_mass["x"][index], centers_of_mass["y"][index]])

    curr_x = centers_of_mass["x"][index]
    curr_y = centers_of_mass["y"][index]
    curr_patch = patch_list[index]
    top_ax.scatter(curr_x, curr_y, s=4)

    if zoom_out:
        top_ax.set_xlim(img_xmin, img_xmax)
        top_ax.set_ylim(img_ymin, img_ymax)
    else:
        xmin, xmax, ymin, ymax = curr_x - 30, curr_x + 30, curr_y - 30, curr_y + 30
        top_ax.set_xlim(xmin, xmax)
        top_ax.set_ylim(ymax, ymin)  # min and max values reversed because in our background image y-axis is reversed

    # Write as x label the speed of the worm
    top_ax.set_xlabel("Speed of the worm: "+str(speed_list[index]))

    top_ax.set_title("Frame: " + str(fd.load_frame(folder, index)) + ", patch: " + str(curr_patch)+", worm xy: "+str(curr_x)+", "+str(curr_y))

    curr_fig = plt.gcf()
    curr_fig.canvas.draw()




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

#avg_intensities1 = np.zeros(len(intensities1))
#avg_intensities2 = np.zeros(len(intensities2))
#avg_intensities3 = np.zeros(len(intensities3))
#avg_intensities4 = np.zeros(len(intensities4))
#avg_intensities_whole_plate = np.zeros(len(intensities1))

#for i in range(min(len(avg_intensities1), len(avg_intensities4))):
    #avg_intensities1[i] = np.mean(image_list[4*i][0:500, 0:500])
    #avg_intensities2[i] = np.mean(image_list[4*i + 1][0:500, -500:-1])
    #avg_intensities3[i] = np.mean(image_list[4*i + 2][-500:-1, -500:-1])
    #avg_intensities4[i] = np.mean(image_list[4*i + 3][-500:-1, 0:500])
    #avg_intensities_whole_plate = np.mean([intensities1[i], intensities2[i], intensities3[i], intensities4[i]])

#fig, axs = plt.subplots(1, 2)

# plt.plot(range(len(intensities1)), intensities1, label="quarter1")
# plt.plot(range(len(intensities2)), intensities2, label="quarter2")
# plt.plot(range(len(intensities3)), intensities3, label="quarter3")
# plt.plot(range(len(intensities4)), intensities4, label="quarter4")
# plt.title("Normalized intensity evolution of each quarter")
# plt.legend()

#axs[1].imshow(image_list[-1])
#axs[1].set_title("Patch aspect at the end")

