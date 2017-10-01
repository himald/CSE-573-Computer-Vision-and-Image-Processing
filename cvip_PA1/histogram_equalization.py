import cv2
import numpy as np
import matplotlib.pyplot as plt

'''------------------############################PROBLEM 2############################------------------'''
#read the image as a grayscale
gray_img = cv2.imread("monalisa.png", 0)

#get dimensions of the image
(img_height, img_width) = gray_img.shape[:2]

#initialize variables for transformation array
pixel_count = img_height * img_width
ratio = 255 / pixel_count

#initialize histogram array
hist_array = np.zeros([256])

#load pixel intensities into histogram array
for y in np.arange(0, img_height):
    for x in np.arange(0, img_width):
        hist_array[gray_img[y, x]] += 1

#plot the histogram
bar_width = 1.3
positions = np.arange(hist_array.size)
plt.xticks(range(0, hist_array.size, 50))
plt.title("Original image histogram")
plt.bar(positions, hist_array, bar_width)
plt.show()

#initialize cumulative histogram array and transformation array
cumul_hist_array = np.zeros([256])
transformation_array = np.zeros([256])

#first index is same for cumulative histogram array and histogram array
cumul_hist_array[0] = hist_array[0]
#calculate value for transformation array's first index
transformation_array[0] = round(ratio * cumul_hist_array[0])

#load cumulative pixel intensities into cumulative histogram array
for i in range(1, hist_array.size):
    cumul_hist_array[i] = cumul_hist_array[i - 1] + hist_array[i]
    transformation_array[i] = round(ratio * cumul_hist_array[i])

#rescan image for transformation
transformed_img = cv2.imread("monalisa.png", 0)

#transform the image using the transformation array
for y in np.arange(0, img_height):
    for x in np.arange(0, img_width):
        transformed_img[y, x] = transformation_array[gray_img[y, x]]

# initialize transformed histogram array
trans_hist_array = np.zeros([256])

# load pixel intensities of transformed image into transformed histogram array
for y in np.arange(0, img_height):
    for x in np.arange(0, img_width):
        trans_hist_array[transformed_img[y, x]] += 1


#plot the cumulative histogram
bar_width = 1.3
positions = np.arange(cumul_hist_array.size)
plt.xticks(range(0, cumul_hist_array.size, 50))
plt.title("Original image cumulative histogram")
plt.bar(positions, cumul_hist_array, bar_width)
plt.show()

#plot the transformation function
plt.title("Transformation Function")
plt.plot(transformation_array)
plt.show()

#plot the transformed image histogram
bar_width = 1.3
positions = np.arange(trans_hist_array.size)
plt.xticks(range(0, trans_hist_array.size, 50))
plt.title("Transformed image histogram")
plt.bar(positions, trans_hist_array, bar_width)
plt.show()

#display the original image and transformed image
cv2.imshow("original_img", gray_img)
cv2.waitKey(0)
cv2.imshow("transformed_img", transformed_img)
cv2.waitKey(0)