import cv2
import numpy as np
import pickle

#read it as a grayscale image

left_view_img = cv2.imread('view1.png', 0)
right_view_img = cv2.imread('view5.png', 0)
ground_truth_left_img = cv2.imread('disp1.png', 0)
ground_truth_right_img = cv2.imread('disp5.png', 0)
(view_img_height, view_img_width) = left_view_img.shape[:2]

left_disparity_matrix_3_win = np.zeros([view_img_height,view_img_width], np.uint8)
left_disparity_matrix_9_win = np.zeros([view_img_height,view_img_width], np.uint8)
right_disparity_matrix_3_win = np.zeros([view_img_height,view_img_width], np.uint8)
right_disparity_matrix_9_win = np.zeros([view_img_height,view_img_width], np.uint8)



#Disparity Computation for Left Image with 3X3 window

#apply padding of 1 unit
left_view_img_1_pad = cv2.copyMakeBorder(left_view_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255, 255))
right_view_img_1_pad = cv2.copyMakeBorder(right_view_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255, 255))

min = 0
min_index = 0
max_disparity = 74

#iterate through each pixel in left image, obtain the window for the pixel and find the best matching pixel in right image
for left_y in range(1, view_img_height + 1):
    for left_x in range(1, view_img_width + 1):
        #print("left_y = " + str(left_y) + " left_x = " + str(left_x))
        #window for left image
        left_window = left_view_img_1_pad[left_y - 1:left_y + 2, left_x - 1:left_x + 2]

        # set the upper limit for iteration in right image row to current left pixel index - max_disparity until it goes below 0
        right_img_upper_limit = 0
        if(left_x > max_disparity):
            right_img_upper_limit = left_x - max_disparity

        #iterate through the same row in right image, move towards left and find pixel value difference at each pixel
        for right_x in range(left_x, right_img_upper_limit, -1):
            # window for right image
            right_window = right_view_img_1_pad[left_y - 1: left_y + 2, right_x - 1:right_x + 2]

            # find the sum squared error between the left and right window
            ssd = np.sum(np.square(left_window - right_window))
            if(right_x == left_x or min > ssd):
                min = ssd
                min_index = right_x

        #best matching pixel is the one with min error

        #print("best_right_pixel = " + str(min_index))
        pixel_disparity = left_x - min_index
        left_disparity_matrix_3_win[left_y - 1, left_x - 1] = pixel_disparity

mse = np.mean((left_disparity_matrix_3_win - ground_truth_left_img)**2)

print("mse_left_3_win = " + str(mse))

cv2.imwrite("stereo_vision_left_disp_3_win.png", left_disparity_matrix_3_win)


#Disparity Computation for Left Image with 9X9 window

#apply padding of 4 units

left_view_img_4_pad = cv2.copyMakeBorder(left_view_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255,255))
right_view_img_4_pad = cv2.copyMakeBorder(right_view_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,255,255,255))

#iterate through each pixel in left image, obtain the window for the pixel and find the best matching pixel in right image
for left_y in range(4, view_img_height + 4):
    for left_x in range(4, view_img_width + 4):
        #print("left_y = " + str(left_y) + " left_x = " + str(left_x))
        # window for left image
        left_window = left_view_img_4_pad[left_y - 4:left_y + 5, left_x - 4:left_x + 5]

        # set the upper limit for iteration in right image row to current left pixel index - max_disparity until it goes below 0
        right_img_upper_limit = 3
        if(left_x > (max_disparity + 3)):
            right_img_upper_limit = left_x - max_disparity

        # iterate through the same row in right image, move towards left and find disparity at each pixel
        for right_x in range(left_x, right_img_upper_limit, -1):
            # window for right image
            right_window = right_view_img_4_pad[left_y - 4: left_y + 5, right_x - 4:right_x + 5]

            # find the sum squared error between the left and right window
            ssd = np.sum(np.square(left_window - right_window))
            if (right_x == left_x or min > ssd):
                min = ssd
                min_index = right_x

        #best matching pixel is the one with min error

        #print("best_right_pixel = " + str(min_index))
        pixel_disparity = left_x - min_index
        left_disparity_matrix_9_win[left_y - 4, left_x - 4] = pixel_disparity

mse = np.mean((left_disparity_matrix_9_win - ground_truth_left_img)**2)

print("mse_left_9_win = " + str(mse))

cv2.imwrite("stereo_vision_left_disp_9_win.png", left_disparity_matrix_9_win)


#Disparity Computation for Right Image with 3X3 window

min = 0
min_index = 0

#iterate through each pixel in right image, obtain the window for the pixel and find the best matching pixel in left image
for right_y in range(1, view_img_height + 1):
    for right_x in range(1, view_img_width + 1):
        #print("right_y = " + str(right_y) + " right_x = " + str(right_x))
        #window for right image
        right_window = right_view_img_1_pad[right_y - 1:right_y + 2, right_x - 1:right_x + 2]

        #set the upper limit for iteration in left image row to current right pixel index + max_disparity until it exceeds image width
        left_img_upper_limit = right_x + max_disparity
        if(left_img_upper_limit > view_img_width):
            left_img_upper_limit = view_img_width


        #iterate through the same row in left image, move towards right and find pixel value difference at each pixel
        for left_x in range(right_x, left_img_upper_limit):
            #window for left image
            left_window = left_view_img_1_pad[right_y - 1:right_y + 2, left_x - 1:left_x + 2]

            # find the sum squared error between the left and right window
            ssd = np.sum(np.square(left_window - right_window))
            if(right_x == left_x or min > ssd):
                min = ssd
                min_index = left_x

        # best matching pixel is the one with min error

        #print("best_left_pixel = " + str(min_index))
        pixel_disparity = min_index - right_x
        right_disparity_matrix_3_win[right_y - 1, right_x - 1] = pixel_disparity


mse = np.mean((right_disparity_matrix_3_win - ground_truth_right_img)**2)

print("mse_right_3_win = " + str(mse))

cv2.imwrite("stereo_vision_right_disp_3_win.png", right_disparity_matrix_3_win)




#Disparity Computation for Right Image with 9X9 window

#iterate through each pixel in right image, obtain the window for the pixel and find the best matching pixel in left image
for right_y in range(4, view_img_height + 4):
    for right_x in range(4, view_img_width + 4):
        #print("right_y = " + str(right_y) + " right_x = " + str(right_x))
        # window for right image
        right_window = right_view_img_4_pad[right_y - 4:right_y + 5, right_x - 4:right_x + 5]

        #set the upper limit for iteration in left image row to current right pixel index + max_disparity until it exceeds image width
        left_img_upper_limit = right_x + max_disparity
        if(left_img_upper_limit > view_img_width):
            left_img_upper_limit = view_img_width

        #iterate through the same row in left image, move towards right and find pixel value difference at each pixel
        for left_x in range(right_x, left_img_upper_limit):
            # window for left image
            left_window = left_view_img_4_pad[right_y - 4: right_y + 5, left_x - 4:left_x + 5]

            # find the sum squared error between the left and right window
            ssd = np.sum(np.square(left_window - right_window))
            if (right_x == left_x or min > ssd):
                min = ssd
                min_index = left_x

        #best matching pixel is the one with min error

        #print("best_left_pixel = " + str(min_index))
        pixel_disparity = min_index - right_x
        right_disparity_matrix_9_win[right_y - 4, right_x - 4] = pixel_disparity

mse = np.mean((right_disparity_matrix_9_win - ground_truth_right_img)**2)

print("mse_right_9_win = " + str(mse))

cv2.imwrite("stereo_vision_right_disp_9_win.png", right_disparity_matrix_9_win)





###############consistency check####################





#consistency check for left disparity matrix obtained with 3X3 window
consistent_left_disparity_matrix_3_win = np.zeros([view_img_height,view_img_width], np.uint8)

#form the consistent left disparity matrix by comparing the disparity values
for i in range(view_img_height):
    for j in range(view_img_width):
        if(j - right_disparity_matrix_3_win[i][j] >= 0 and left_disparity_matrix_3_win[i][j] == right_disparity_matrix_3_win[i][j - left_disparity_matrix_3_win[i][j]]):
            consistent_left_disparity_matrix_3_win[i][j] = left_disparity_matrix_3_win[i][j]

squared_error = np.zeros([view_img_height, view_img_width], np.uint8)
error_sum = 0
consistent_pixel_count = 0

#calculate mean squared error for consistent pixels
for i in range(view_img_height):
    for j in range(view_img_width):
        if(consistent_left_disparity_matrix_3_win[i][j] != 0):
            squared_error[i][j] = (consistent_left_disparity_matrix_3_win[i][j] - ground_truth_left_img[i][j])**2
            error_sum += squared_error[i][j]
            consistent_pixel_count += 1

mse = error_sum / consistent_pixel_count
print("mse_consistent_left_3_win = " + str(mse))
cv2.imwrite("stereo_vision_consistent_left_disp_3_win.png", consistent_left_disparity_matrix_3_win)


#consistency check for left disparity matrix obtained with 9X9 window
consistent_left_disparity_matrix_9_win = np.zeros([view_img_height,view_img_width], np.uint8)

#form the consistent left disparity matrix by comparing the disparity values
for i in range(view_img_height):
    for j in range(view_img_width):
        if(j - right_disparity_matrix_3_win[i][j] >= 0 and left_disparity_matrix_9_win[i][j] == right_disparity_matrix_9_win[i][j - left_disparity_matrix_9_win[i][j]]):
            consistent_left_disparity_matrix_9_win[i][j] = left_disparity_matrix_9_win[i][j]

squared_error = np.zeros([view_img_height, view_img_width], np.uint8)
error_sum = 0
consistent_pixel_count = 0

#calculate mean squared error for consistent pixels
for i in range(view_img_height):
    for j in range(view_img_width):
        if(consistent_left_disparity_matrix_9_win[i][j] != 0):
            squared_error[i][j] = (consistent_left_disparity_matrix_9_win[i][j] - ground_truth_left_img[i][j])**2
            error_sum += squared_error[i][j]
            consistent_pixel_count += 1

mse = error_sum / consistent_pixel_count
print("mse_consistent_left_9_win = " + str(mse))
cv2.imwrite("stereo_vision_consistent_left_disp_9_win.png", consistent_left_disparity_matrix_9_win)



#consistency check for right disparity matrix obtained with 3X3 window
consistent_right_disparity_matrix_3_win = np.zeros([view_img_height,view_img_width], np.uint8)

#form the consistent right disparity matrix by comparing the disparity values
for i in range(view_img_height):
    for j in range(view_img_width):
        if(j + right_disparity_matrix_3_win[i][j] < view_img_width and right_disparity_matrix_3_win[i][j] == left_disparity_matrix_3_win[i][j + right_disparity_matrix_3_win[i][j]]):
            consistent_right_disparity_matrix_3_win[i][j] = right_disparity_matrix_3_win[i][j]


squared_error = np.zeros([view_img_height, view_img_width], np.uint8)
error_sum = 0
consistent_pixel_count = 0

#calculate mean squared error for consistent pixels
for i in range(view_img_height):
    for j in range(view_img_width):
        if(consistent_right_disparity_matrix_3_win[i][j] != 0):
            squared_error[i][j] = (consistent_right_disparity_matrix_3_win[i][j] - ground_truth_right_img[i][j])**2
            error_sum += squared_error[i][j]
            consistent_pixel_count += 1


mse = error_sum / consistent_pixel_count
print("mse_consistent_right_3_win = " + str(mse))
cv2.imwrite("stereo_vision_consistent_right_disp_3_win.png", consistent_right_disparity_matrix_3_win)



#consistency check for right disparity matrix obtained with 9X9 window
consistent_right_disparity_matrix_9_win = np.zeros([view_img_height,view_img_width], np.uint8)

#form the consistent right disparity matrix by comparing the disparity values
for i in range(view_img_height):
    for j in range(view_img_width):
        if(j + right_disparity_matrix_9_win[i][j] < view_img_width and right_disparity_matrix_9_win[i][j] == left_disparity_matrix_9_win[i][j + right_disparity_matrix_9_win[i][j]]):
            consistent_right_disparity_matrix_9_win[i][j] = right_disparity_matrix_9_win[i][j]


squared_error = np.zeros([view_img_height, view_img_width], np.uint8)
error_sum = 0
consistent_pixel_count = 0

#calculate mean squared error for consistent pixels
for i in range(view_img_height):
    for j in range(view_img_width):
        if(consistent_right_disparity_matrix_9_win[i][j] != 0):
            squared_error[i][j] = (consistent_right_disparity_matrix_9_win[i][j] - ground_truth_right_img[i][j])**2
            error_sum += squared_error[i][j]
            consistent_pixel_count += 1


mse = error_sum / consistent_pixel_count
print("mse_consistent_right_9_win = " + str(mse))
cv2.imwrite("stereo_vision_consistent_right_disp_9_win.png", consistent_right_disparity_matrix_9_win)



filehandler = open("consistent_left_disparity_matrix_9_win.obj","wb")
pickle.dump(consistent_left_disparity_matrix_9_win,filehandler)
filehandler.close()

filehandler = open("consistent_right_disparity_matrix_9_win.obj","wb")
pickle.dump(consistent_right_disparity_matrix_9_win,filehandler)
filehandler.close()
