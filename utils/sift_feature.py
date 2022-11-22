'''
Author: Peng Bo
Date: 2022-11-22 14:35:12
LastEditTime: 2022-11-22 16:30:15
Description: 

'''
import numpy as np
import cv2 as cv2
import pdb

def sift_feat(img):
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    pts_2d, descriptors = sift.detectAndCompute(gray,None)
    # pts_2d = [element.pt for element in pts_2d]
    descriptors  = descriptors.tolist()
    return pts_2d, descriptors

def filter_sift_descriptors(img, box):
    pts_2d, descriptors = sift_feat(img)
    filter_pts_2d, filter_descriptors = [], []

    def _is_in_box_(x, y):
        if (x>box[0] and x<box[2]) and (y>box[1] and y<box[3]):
            return True
        else:
            return False
    for pt2d, desp in zip(pts_2d, descriptors):
        if not _is_in_box_(pt2d.pt[0], pt2d.pt[1]):
            filter_pts_2d.append(pt2d)
            filter_descriptors.append(desp)
    
    return filter_pts_2d, np.array(filter_descriptors).astype(np.float32)


def desp_match(desps_1, desps_2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desps_1, desps_2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    return matchesMask


def my_match(img1, img2, kp1, kp2, des1, des2, mc_type):
    #不难发现，m,n就是对应刚才的2
    for m,n in matches:
        if m.distance < 0.95 *n.distance:
            print(m.distance)
            print(m.queryIdx)
            print('---end---')
            # print(n.imgIdx)
            good.append(m)
    if len(good) > 10:
        src_pts = np.float32([sift_kp_1[i.queryIdx].pt for i in good])#.\reshape(-1, 1, 2)
        dst_pts = np.float32([sift_kp_2[i.trainIdx].pt for i in good])#.reshape(-1, 1, 2)
        # print(src_pts.shape)
        # print(dst_pts.shape)
        ransacReprojThreshold = 10.0
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
        matchesMask = mask.ravel().tolist()
        h, w, mode = img1.shape
        pts = np.float32([[0, 0], [0, h -1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #透视变换函数cv2.perspectiveTransform: 输入的参数是两种数组，并返回dst矩阵——扭转矩阵
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (127,255,0), 3, cv2.LINE_AA)



def is_background_move(desps_1, desps_2):
    matchesMask = desp_match(desps_1, desps_2) 


def get_avg_distance(pts2d_1, desps_1, pts2d_2, desps_2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desps_1, desps_2, k=2)
    valid_matches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)

    src_pts = np.float32([pts2d_1[i.queryIdx].pt for i in valid_matches])
    dst_pts = np.float32([pts2d_2[i.trainIdx].pt for i in valid_matches])
    avg_distance = np.linalg.norm(src_pts - dst_pts, ord=1)
    return avg_distance

def sift_match(img_1, img_2, box_1, box_2):
    pts2d_1, desps_1 = filter_sift_descriptors(img_1, box_1)
    pts2d_2, desps_2 = filter_sift_descriptors(img_2, box_2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desps_1, desps_2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0), 
                    matchesMask=matchesMask,
                    flags=0)
    
    flannmaches = cv2.drawMatchesKnn(img_1, pts2d_1, img_2, pts2d_2, matches, None, **draw_params)
    return flannmaches, pts2d_1, desps_1, pts2d_2, desps_2


if __name__ == "__main__":
    img1 = cv2.imread('data/test.jpg')
    img2 = cv2.imread('data/test.jpg')
    box1 = [20, 20, 30, 30]
    box2 = [20, 20, 30, 30]
    flannmaches, pts2d_1, desps_1, pts2d_2, desps_2 = sift_match(img1, img2, box1, box2)

    avg_distance = get_avg_distance(pts2d_1, desps_1, pts2d_2, desps_2)

    cv2.imshow('match result', flannmaches)
    key = cv2.waitKey(-1)
    if key == 27:
        exit(1)
    import pdb
    pdb.set_trace()