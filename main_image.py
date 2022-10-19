# Importing libraries
from turtle import color
import numpy as np
import cv2
from matplotlib import pyplot as plt

match_num = 2

# Reading images
template = cv2.imread('template2.png', 0)
original = cv2.imread('org1.jpeg', 0)

# Making histograms equal
template_eq_hist = cv2.equalizeHist(template)
original_eq_hist = cv2.equalizeHist(original)

# Creating SIFT detector
sift = cv2.SIFT_create()

# Finding keypoints and descriptors
keyp1, desc1 = sift.detectAndCompute(template_eq_hist, None)
keyp2, desc2 = sift.detectAndCompute(original_eq_hist, None)

# Finding keypoints and descriptors
flann_index_kdtree = 0
indexP = dict(algorithm=flann_index_kdtree, trees=5)
searchP = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexP, searchP)

# Finding matcher by using knnMatch
matches = flann.knnMatch(desc1, desc2, k=2)

# Storing all the suitable matches
suitable_matches = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        suitable_matches.append(m)

if len(suitable_matches) > match_num:
    src_pts = np.float32(
        [keyp1[m.queryIdx].pt for m in suitable_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keyp2[m.trainIdx].pt for m in suitable_matches]).reshape(-1, 1, 2)

    # Finding homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = template_eq_hist.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                        [w-1, 0]]).reshape(-1, 1, 2)
    
    # Matched coordinates
    dst = cv2.perspectiveTransform(pts, M) 
        
    map_img = cv2.polylines(
        original_eq_hist, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" %
            (len(suitable_matches), match_num))
    matchesMask = None
# Drawing matches in green color
draw_params = dict(matchColor=(0, 255, 0),  
                    singlePointColor=None,
                    matchesMask=matchesMask, 
                    flags=2)

# Drawing final match points
img3 = cv2.drawMatches(template_eq_hist, keyp1, map_img, keyp2,
                        suitable_matches, None, **draw_params)

# Let's show final result
plt.imshow(img3, 'gray'), plt.show()

