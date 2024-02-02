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

# Time steps at which to start / end analysis
start_time = 1237
end_time = min(3000, len(image_path_list))

# For now only pick one image out of n_lines x n_columns, to not worry with stitching or whatever
image_list = []
for i in range(start_time, end_time):
    if i % (n_lines * n_columns) == 0:
        image = cv2.imread(image_path_list[i], -1)  # -1 parameter is to not ruin the .tif upon import!!! (bit depth is 16)
        #image = cv2.addWeighted(image, 60, np.zeros(image.shape, image.dtype), 0, 100)
        image_list.append(image)

beginning_avg_image = np.mean([image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], image_list[6], image_list[7], image_list[8], image_list[9]], axis=0)  # average across arrays of the list
end_avg_image = np.mean([image_list[-1], image_list[-2], image_list[-3], image_list[-4], image_list[-5], image_list[-6], image_list[-7], image_list[-8], image_list[-9], image_list[-10]], axis=0)

fig, axs = plt.subplots(1, 3, layout="constrained")

axs[0].imshow(beginning_avg_image, vmin=400, vmax=900)
axs[0].set_title("Image around t="+str(start_time))

axs[1].imshow(end_avg_image, vmin=400, vmax=900)
axs[1].set_title("Image around t="+str(end_time))

plot = axs[2].imshow(end_avg_image - beginning_avg_image + 1000)
axs[2].set_title("Difference between the two times")

fig.colorbar(plot, ax=axs[0:3], location="bottom", shrink=0.5)
plt.show()

