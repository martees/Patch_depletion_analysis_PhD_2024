import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# Paths to find images
path = "/media/admin/LaCie/Data/MicroMonitWorm/depletion_pilots/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"
image_path_list = sorted(glob.glob(path + "*.tif", recursive=False))

# Number of lines and columns to the microscope tiling (for when you take only a quarter of the drop on each img for ex)
n_lines = 2
n_columns = 2

# Which quarter to look at (0 = upper left, 1 = upper right, 2 = lower right, 3 = lower left)
quarter = 3

# Time steps at which to start / end analysis
start_time = 400
end_time = min(3000, len(image_path_list))

# To avoid noise, images in the beginning and end are averaged over a certain number of frames: define here
average_depth = 10

# Go through the list of images to read them
image_list = []
for i in range(start_time, end_time):
    if i % (n_lines * n_columns) == quarter:
        image = cv2.imread(image_path_list[i],
                           -1)  # -1 parameter is to not ruin the .tif upon import!!! (bit depth is 16)
        # image = cv2.addWeighted(image, 60, np.zeros(image.shape, image.dtype), 0, 100)
        image_list.append(image)

# Make an average version for the beginning and the end (averaged over :average_depth: frames)
beginning_images = []
end_images = []
for i in range(average_depth):
    beginning_images.append(image_list[i])
    end_images.append(image_list[-i-1])

beginning_avg_image = np.mean(beginning_images, axis=0)  # average across arrays of the list
end_avg_image = np.mean(end_images, axis=0)

# Make two images that are also averages over :average_depth: but on a corner of the image where there's no worm/food
# Define corner bounds based on image number (quarter 0 is upper left image so the upper left corner should be empty)
if quarter == 0:  # upper left
    empty_lower_x = 0
    empty_upper_x = 500
    empty_lower_y = 0
    empty_upper_y = 500
if quarter == 1:  # upper right [0:500, -500:-1]
    empty_lower_x = 0
    empty_upper_x = 500
    empty_lower_y = -500
    empty_upper_y = -1
if quarter == 2:  # bottom right [-500:-1, -500:-1]
    empty_lower_x = -500
    empty_upper_x = -1
    empty_lower_y = -500
    empty_upper_y = -1
if quarter == 3:  # bottom left [-500:-1, 0:500]
    empty_lower_x = -500
    empty_upper_x = -1
    empty_lower_y = 0
    empty_upper_y = 500

# Just take the previous image lists but only the empty part
beginning_empty_corner = [beginning_images[i][empty_lower_x:empty_upper_x, empty_lower_y:empty_upper_y] for i in range(len(beginning_images))]
end_empty_corner = [end_images[i][empty_lower_x:empty_upper_x, empty_lower_y:empty_upper_y] for i in range(len(end_images))]
beginning_avg_empty_corner = np.mean(beginning_empty_corner)
end_avg_empty_corner = np.mean(end_empty_corner)


fig, axs = plt.subplots(1, 4, layout="constrained")

axs[0].imshow(beginning_avg_image/beginning_avg_empty_corner, vmin=1, vmax=1.3)
axs[0].set_title("Normalized image around t=" + str(start_time))

axs[1].imshow(end_avg_image/end_avg_empty_corner, vmin=1, vmax=1.3)
axs[1].set_title("Normalized image around t=" + str(end_time))

# plot = axs[2].imshow((end_avg_image - beginning_avg_image)/(end_avg_empty_corner - beginning_avg_empty_corner))
# axs[2].set_title("Normalized difference between the two images")

plot = axs[2].imshow(end_avg_image/end_avg_empty_corner - beginning_avg_image/beginning_avg_empty_corner, vmin=-1, vmax=0)
axs[2].set_title("Difference between the two normalized images: negative part (-1 - 0)")

plot = axs[3].imshow(end_avg_image/end_avg_empty_corner - beginning_avg_image/beginning_avg_empty_corner, vmin=0, vmax=0.2)
axs[3].set_title("Difference between the two normalized images: positive part (0 - 0.2)")

fig.colorbar(plot, ax=axs[0:4], location="bottom", shrink=0.5)
plt.show()
