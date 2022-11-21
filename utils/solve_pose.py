'''
Author: Peng Bo
Date: 2022-11-21 14:33:35
LastEditTime: 2022-11-21 21:05:34
Description: 

'''
import cv2
import numpy as np
import pdb

points_3d = np.array([
    [32.788307 , -38.764736 ,  -0.8804663],
    [-32.934914 , -38.286953 ,  -2.105181 ],
    [  0.55592  ,   6.5629   ,  25.944448 ],
    [-28.916267 ,  28.612717 ,   2.24031  ],
    [ 28.794413 ,  28.079924 ,   3.217393 ]
], dtype = np.float64)

points3p_3d = np.array([
    [32.788307 , -38.764736 ,  -0.8804663],
    [-32.934914 , -38.286953 ,  -2.105181 ],
    [  0.55592  ,   6.5629   ,  25.944448 ]
], dtype = np.float64)


focal_length = 400

def pose_estimate(pts_2d, pts_3d=points_3d, img_size=(180, 320), dist_coeffs=np.zeros((4, 1))):
    # focal_length = max(img_size[0], img_size[1])
    camera_center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, camera_center[0]],
        [0, focal_length, camera_center[1]],
        [0, 0, 1]
    ], dtype=np.double)
    # _, rot_vec, trt_vec = cv2.solvePnP(pts_3d, pts_2d.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

    if pts_2d.shape[0] == 3:
        nose = pts_2d[2]
        right = pts_2d[1]
        left = pts_2d[1]
        dist1 = np.linalg.norm(nose - right)
        dist2 = np.linalg.norm(nose - left)
        if dist1 <= dist2:
            r_vec = np.array([[0.34554543], [-0.72173726], [0.08495318]])
            t_vec = np.array([[-12.14525577], [-48.03475936], [383.82047981]])
        else:
            r_vec = np.array([[0.75807009], [0.3207348], [-2.80691676]])
            t_vec = np.array([[-24.07046963], [-1.68285571], [-199.17583135]])
        _, rot_vec, trt_vec = cv2.solvePnP(points3p_3d, pts_2d, camera_matrix, dist_coeffs, 
                    rvec=r_vec, tvec=t_vec, useExtrinsicGuess=True, 
                    flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        _, rot_vec, trt_vec, _ = cv2.solvePnPRansac(pts_3d, pts_2d.astype(np.float64), 
                    camera_matrix, dist_coeffs)
        #_, rot_vec, trt_vec, _ = cv2.solvePnPRansac(pts_3d, pts_2d.astype(np.float64), 
        #            camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    return rot_vec, trt_vec


if __name__ == "__main__":
    pts_2d = [
        
    ]
    rot_vec, trt_vec = pose_estimate(pts_2d, img_size=(240, 320))