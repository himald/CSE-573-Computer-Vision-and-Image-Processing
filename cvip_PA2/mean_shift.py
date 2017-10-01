import cv2
import numpy as np

#load the image
input_img = cv2.imread('Butterfly.jpg')
(input_img_height, input_img_width) = input_img.shape[:2]

output_img = np.zeros([input_img_height, input_img_width, 3], np.uint8)


#convert the image into dataset of R, G, B, x, y
dataset = np.empty([input_img_height * input_img_width, 5], dtype=np.int)


row = 0
for y in range(input_img_height):
    for x in range(input_img_width):
        dataset[row][0] = input_img[y][x][0]
        dataset[row][1] = input_img[y][x][1]
        dataset[row][2] = input_img[y][x][2]
        dataset[row][3] = y
        dataset[row][4] = x
        row += 1

print ("dataset ready")

#initialize h (cluster expansion threshold) and iter(mean shift threshold)
h = 90
iter = 35



count = 0
while(dataset.shape[0] > 0):
    print("iteration " + str(count) + ": " + str(dataset.shape[0]))
    count += 1
    #select a random mean
    random_mean_index = np.random.randint(0, dataset.shape[0])
    random_mean = dataset[random_mean_index]

    #initialize the cluster matrix
    cluster = np.zeros([input_img_height * input_img_width, 5], dtype=np.int)

    #add mean to cluster
    cluster[random_mean_index] = random_mean

    discardable_indexes_marker_array = np.zeros([dataset.shape[0]])

    for data_point_index in range(dataset.shape[0]):
        data_point = dataset[data_point_index]

        #calculate euclidean distance of data_point from the random mean
        distance = np.sqrt(np.sum(np.square(random_mean - data_point)))

        #if distance < h then add it to cluster
        if(distance < h):
            cluster[data_point_index] = data_point
            discardable_indexes_marker_array[data_point_index] = 1

    cluster_non_zero = cluster[[i for i, x in enumerate(cluster) if x.any()]]

    cluster_mean = np.mean(cluster_non_zero, axis=0)
    mean_distance = np.sqrt(np.sum(np.square(random_mean - cluster_mean)))

    #if distance from mean is less than mean shift threshold then use the cluster for output image and delete its entries from dataset
    if(mean_distance < iter):
        for output_point_index in range(cluster_non_zero.shape[0]):
            output_img[cluster_non_zero[output_point_index][3]][cluster_non_zero[output_point_index][4]][0] = random_mean[0]
            output_img[cluster_non_zero[output_point_index][3]][cluster_non_zero[output_point_index][4]][1] = random_mean[1]
            output_img[cluster_non_zero[output_point_index][3]][cluster_non_zero[output_point_index][4]][2] = random_mean[2]

        discardable_indexes = np.nonzero(discardable_indexes_marker_array)[0]
        dataset = np.delete(dataset, discardable_indexes, 0)
cv2.imwrite("mean_shift_semgented_image_h_90.jpg", output_img)

