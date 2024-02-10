import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

image_path = "/media/admin/LaCie/Data/MicroMonitWorm/depletion_pilots/20243101_OD0.2oldbact10%gfp4s_Lsomethingworm_dishupsidedown-02/"
image_path_list = sorted(glob.glob(image_path + "*.tif", recursive=False))

image_list = []
for i_image in range(len(image_path_list)):
    if i_image % 4 == 2:
        image = cv2.imread(image_path_list[i_image], -1)
        # Convert image to 8 bit depth
        ratio = np.amax(image) / 256
        image = (image / ratio).astype('uint8')
        image_list.append(image)

curr_frame = 140

last_5_frames = np.mean([image_list[i] for i in range(curr_frame - 5, curr_frame)], axis=0)
last_5_diff = image_list[curr_frame] - last_5_frames

_, image_thresh = cv2.threshold(last_5_diff, np.quantile(last_5_diff, 0.9), 255, cv2.THRESH_BINARY)
# num_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresh)
kernel = np.ones((5, 5), np.uint8)
image_denoise = cv2.erode(image_thresh, kernel, iterations=1)
image_denoise = cv2.dilate(image_denoise, kernel, iterations=7)

fig, axs = plt.subplots(1, 2)

axs[0].imshow(last_5_diff)
axs[1].imshow(image_denoise)
#axs[0].scatter([centroids[i][0] for i in range(len(centroids))], [centroids[i][1] for i in range(len(centroids))])

plt.show()


#
# axs[0, 0].imshow(image_list[curr_frame], vmin=0)
# axs[0, 0].set_title("image_list[curr_frame]")
#
# axs[0, 1].imshow(last_20_diff, vmin=0)
# axs[0, 1].set_title("last_20_diff")
#
# axs[1, 0].imshow(last_10_diff, vmin=0)
# axs[1, 0].set_title("last_10_diff")
#
# axs[1, 1].imshow(last_5_diff, vmin=0)
# axs[1, 1].set_title("last_5_frames")


# _, image_thresh = cv2.threshold(image, np.quantile(image, 0.9), 255, cv2.THRESH_BINARY)
# kernel = np.ones((5, 5), np.uint8)
# image_denoise = cv2.erode(image_thresh, kernel, iterations=1)
# image_denoise = cv2.dilate(image_denoise, kernel, iterations=13)
# num_blobs, labels, stats, centroids = cv2.connectedComponentsWithStats(image_denoise)
#
# # Width, height and area of each blob
# heights = stats[:, cv2.CC_STAT_HEIGHT]
# widths = stats[:, cv2.CC_STAT_WIDTH]
# areas = stats[:, cv2.CC_STAT_AREA]
#
# height2width_ratio = heights / widths

# second_biggest_index = np.argsort(areas)[-3]
# second_biggest_blob = centroids[second_biggest_index]


#left_ax.imshow(image)
#right_ax.imshow(image_denoise)
#right_ax.scatter([centroids[i][0] for i in range(len(centroids))], [centroids[i][1] for i in range(len(centroids))], c=height2width_ratio, cmap="hot")
# right_ax.scatter([second_biggest_blob[0]], [second_biggest_blob[1]], color="red")



#
# for i_centroid in range(len(centroids)):
#     plt.scatter([centroids[i_centroid][0]], [centroids[i_centroid][1]], s=1)
#     area_ratio = int(areas[i_centroid]/avg_bact_size)
#     if area_ratio > 1:
#         plt.text(centroids[i_centroid][0]+1, centroids[i_centroid][1]+1, str(area_ratio), color="white")
# plt.title("There are "+str(len(centroids))+" bacteria here")


#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#scale_percent = 20  # percent of original size

#width = int(frame.shape[1] * scale_percent / 100)
#height = int(frame.shape[0] * scale_percent / 100)
#dim = (width, height)
#frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# frame = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel1)
# _, frame = cv2.threshold(frame, 48, 255, cv2.THRESH_BINARY)
# frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 10)
# frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel2)

#thresh_video.append(frame)