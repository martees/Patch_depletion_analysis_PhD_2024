import os


def is_linux():  # returns true if you're using linux, otherwise false
    try:
        test = os.uname()
        if test[0] == "Linux":
            return True
    except AttributeError:
        return False


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
