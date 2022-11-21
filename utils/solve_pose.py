import cv2
import numpy as np

points_3d = [
    [32.788307 , -38.764736 ,  -0.8804663],
    [-32.934914 , -38.286953 ,  -2.105181 ],
    [  0.55592  ,   6.5629   ,  25.944448 ],
    [-28.916267 ,  28.612717 ,   2.24031  ],
    [ 28.794413 ,  28.079924 ,   3.217393 ]
]

def pose_estimate(pts_2d, pts_3d=points_3d, img_size=(480, 640), dist_coeffs = np.zeros((4, 1))):
    focal_length = max(img_size[0], img_size[1])
    camera_center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, camera_center[0]],
        [0, focal_length, camera_center[1]],
        [0, 0, 1]
    ], dtype=np.double)
    _, rot_vec, trt_vec = cv2.solvePnP(pts_3d, pts_2d, camera_matrix, dist_coeffs)
    return rot_vec, trt_vec


if __name__ == "__main__":
    pts_2d = [
        
    ]
    rot_vec, trt_vec = pose_estimate(pts_2d, img_size=(240, 320))