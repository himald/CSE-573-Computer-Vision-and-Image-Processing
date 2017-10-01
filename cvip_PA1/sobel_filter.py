import cv2
import numpy as np

'''------------------############################PROBLEM 1############################------------------'''
#read the image as a grayscale
img = cv2.imread("lena_gray.png", 0)

#get dimensions of the image
(img_height, img_width) = img.shape[:2]

'''------------------##############PROBLEM 1.1##############------------------'''

#initialize sobel filters in 2D
x_mat = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="int")
y_mat = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="int")

#add padding to image
img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

#initialize the output convoluted matrices and matrices for their normal forms
g_x = np.zeros((img_height, img_width), dtype="float32")
g_x_norm = np.zeros((img_height, img_width), dtype="float32")
g_y = np.zeros((img_height, img_width), dtype="float32")
g_y_norm = np.zeros((img_height, img_width), dtype="float32")
g_norm = np.zeros((img_height, img_width), dtype="float32")

#iterate through each pixel, obtain the window for the pixel and apply the filters to get the updated pixels
for y in np.arange(1, img_height + 1):
    for x in np.arange(1, img_width + 1):
        window = img[y - 1:y + 2, x - 1:x + 2]

        updated_for_g_x = np.sum(np.multiply(window, x_mat))
        updated_for_g_y = np.sum(np.multiply(window, y_mat))

        g_x[y - 1, x - 1] = updated_for_g_x
        g_y[y - 1, x - 1] = updated_for_g_y

#normalize the obtained convoluted matrices such that the intensities range from 0 to 1
cv2.normalize(g_x, g_x_norm, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(g_y, g_y_norm, 0, 1, cv2.NORM_MINMAX)

#combine the outputs obtained from the 2 sobel filters to obtain the edges of the image
g = np.sqrt(np.square(g_x) + np.square(g_y))
cv2.normalize(g, g_norm, 0, 1, cv2.NORM_MINMAX)

#display the resultant images
cv2.imshow("Gx", g_x_norm)
cv2.waitKey(0)
cv2.imshow("Gy", g_y_norm)
cv2.waitKey(0)
cv2.imshow("G", g_norm)
cv2.waitKey(0)


'''------------------##############PROBLEM 1.2##############------------------'''
#initialize sobel filters in 1D
split_x_mat_horz = np.matrix([-1, 0, 1], dtype="int")
split_x_mat_vert = np.matrix([1, 2, 1], dtype="int")
split_y_mat_horz = np.matrix([1, 2, 1], dtype="int")
split_y_mat_vert = np.matrix([-1, 0, 1], dtype="int")

#initialize the output convoluted matrices and matrices for their normal forms
g_x_new = np.zeros((img_height, img_width), dtype="float32")
g_x_norm_new = np.zeros((img_height, img_width), dtype="float32")
g_y_new = np.zeros((img_height, img_width), dtype="float32")
g_y_norm_new = np.zeros((img_height, img_width), dtype="float32")
g_norm_new = np.zeros((img_height, img_width), dtype="float32")


#iterate through each pixel, obtain the window for the pixel and apply the filters to get the updated pixels
for y in np.arange(1, img_height + 1):
    for x in np.arange(1, img_width + 1):
        window_new = img[y - 1:y + 2, x - 1:x + 2]

        updated_for_g_x_new = np.dot(split_x_mat_vert, np.dot(window_new, split_x_mat_horz.T))
        updated_for_g_y_new = np.dot(split_y_mat_vert, np.dot(window_new, split_y_mat_horz.T))

        g_x_new[y - 1, x - 1] = updated_for_g_x_new
        g_y_new[y - 1, x - 1] = updated_for_g_y_new

#normalize the obtained convoluted matrices such that the intensities range from 0 to 1
cv2.normalize(g_x_new, g_x_norm_new, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(g_y_new, g_y_norm_new, 0, 1, cv2.NORM_MINMAX)

#combine the outputs obtained from the 2 sobel filters to obtain the edges of the image
g_new = np.sqrt(np.square(g_x_new) + np.square(g_y_new))
cv2.normalize(g_new, g_norm_new, 0, 1, cv2.NORM_MINMAX)

diff = g_new - g
#display the resultant images
cv2.imshow("Gx_new", g_x_norm_new)
cv2.waitKey(0)
cv2.imshow("Gy_new", g_y_norm_new)
cv2.waitKey(0)
cv2.imshow("G_new", g_norm_new)
cv2.waitKey(0)