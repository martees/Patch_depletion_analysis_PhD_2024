import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = "/home/admin/Desktop/Patch_depletion/Microscope/2023_12_12_OP50GFP_Patches_microscope/"
image = cv2.imread(image_path+"231212_OD0_2_x10_10gfp4s_slide_2-01-Image Export-01.tif")

_, image_thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)
image_thresh = cv2.cvtColor(image_thresh, cv2.COLOR_BGR2GRAY)
_, _, stats, centroids = cv2.connectedComponentsWithStats(image_thresh)
centroids = centroids[1:]
areas = stats[1:, 4]  # we start at 1 to remove first blob which is the background
median_bact_size = np.median(areas)
avg_bact_size = np.mean(areas)

#image[:, :, 0] = image[:, :, 0]*10
#image[:, :, 1] = image[:, :, 1]*10
#image[:, :, 2] = image[:, :, 2]*10

plt.imshow(image)
#plt.scatter([centroids[i][0] for i in range(len(centroids))], [centroids[i][1] for i in range(len(centroids))])
for i_centroid in range(len(centroids)):
    plt.scatter([centroids[i_centroid][0]], [centroids[i_centroid][1]], s=1)
    area_ratio = int(areas[i_centroid]/avg_bact_size)
    if area_ratio > 1:
        plt.text(centroids[i_centroid][0]+1, centroids[i_centroid][1]+1, str(area_ratio), color="white")
plt.title("There are "+str(len(centroids))+" bacteria here")
plt.show()

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