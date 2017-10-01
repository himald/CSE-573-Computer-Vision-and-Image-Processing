import cv2
import numpy as np

left_img = cv2.imread('disp1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('disp5.png', 0)

(numrows, numcols) = left_img.shape[:2]

#Disparity Computation for Left and Right Image

OcclusionCost = 20 #(You can adjust this, depending on how much threshold you want to give for noise)

#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols

CostMatrix = np.zeros([numcols,numcols])
DirectionMatrix = np.zeros([numcols,numcols])  #(This is important in Dynamic Programming. You need to know which direction you need traverse)
left_DisparityMatrix = np.zeros([numrows,numcols], np.uint8)
right_DisparityMatrix = np.zeros([numrows,numcols], np.uint8)

#We first populate the first row and column values of Cost Matrix

for i in range (0, numcols):
    CostMatrix[i, 0] = i * OcclusionCost
    CostMatrix[0, i] = i * OcclusionCost

# Now, its time to populate the whole Cost Matrix and DirectionMatrix

# Use the pseudocode from "A Maximum likelihood Stereo Algorithm" paper given as reference

for row in range(1, numrows):
    print (row)
    for i in range (1, numcols):
        for j in range(1, numcols):
            min1 = CostMatrix[i - 1, j - 1] + np.abs(left_img[row,i] - right_img[row,j])
            min2 = CostMatrix[i - 1, j] + OcclusionCost
            min3 = CostMatrix[i, j - 1] + OcclusionCost
            cmin = min(min1, min2, min3)
            CostMatrix[i, j] = cmin
            if(min1 == cmin):
                DirectionMatrix[i, j] = 1
            if(min2 == cmin):
                DirectionMatrix[i, j] = 2
            if(min3 == cmin):
                DirectionMatrix[i, j] = 3

    p=numcols - 1
    q=numcols - 1
    while(p!=0 and q!=0):
        if(DirectionMatrix[p, q] == 1.0):
            left_DisparityMatrix[row, p] = np.abs(p - q)
            right_DisparityMatrix[row, q] = np.abs(p - q)
            p = p - 1
            q = q - 1
        elif(DirectionMatrix[p, q] == 2.0):
            p = p - 1
        elif(DirectionMatrix[p, q] == 3.0):
            q = q - 1

print ("max_left = " + str(np.max(left_DisparityMatrix)))
cv2.imwrite("left_disp_dynamic_output.png", left_DisparityMatrix)

print ("max_right = " + str(np.max(right_DisparityMatrix)))
cv2.imwrite("right_disp_dynamic_output.png", right_DisparityMatrix)

