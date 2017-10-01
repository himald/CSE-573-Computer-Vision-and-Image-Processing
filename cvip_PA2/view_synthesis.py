import cv2
import numpy as np
import pickle

#load the color images
view1_img = cv2.imread('view1.png')
view5_img = cv2.imread('view5.png')

left_disparity_matrix = cv2.imread('disp1.png', 0)
right_disparity_matrix = cv2.imread('disp5.png', 0)

(view_img_height, view_img_width) = view1_img.shape[:2]

#initialize blank image
view3_img = np.zeros((view_img_height,view_img_width,3), np.uint8)

#synthesize using left image and left disparity matrix obtained with 9X9 window
for i in range(view_img_height):
    for j in range(view_img_width):
        index_shift_left = int(np.floor(left_disparity_matrix[i][j] / 2))
        if(j - index_shift_left >= 0 and left_disparity_matrix[i][j] != 0):
            view3_img[i][j - index_shift_left] = view1_img[i][j]

cv2.imwrite("view3_output.png", view3_img)

#synthesize using right image and right disparity matrix obtained with 9X9 window
for i in range(view_img_height):
    for j in range(view_img_width):
        index_shift_right = int(np.floor(right_disparity_matrix[i][j] / 2))
        if ((j + index_shift_right < view_img_width and view3_img[i][j + index_shift_right][0] == 0 and view3_img[i][j + index_shift_right][1] == 0 and view3_img[i][j + index_shift_right][2] == 0)):
            view3_img[i][j + index_shift_right] = view5_img[i][j]

cv2.imwrite("view3_output_refined.png", view3_img)