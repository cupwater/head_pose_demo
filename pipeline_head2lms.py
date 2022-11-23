'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-11-23 19:41:44
Description: 

'''
# coding: utf8

import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_facelms_v2 import detect_facelms_v2
from utils.solve_pose import pose_estimate, trt_vec2height
from utils.head2body_box import head2body_box1
from utils.sift_feature import filter_sift_descriptors, get_avg_distance
from utils.VirtualDesk import VirtualDesk
from utils.myqueue import MyQueue

import pdb

adjust_signal = False
last_desp, last_pts2d = None, None
img_size = (540, 960)
detect_interval = 10
move_threshold = 2

def _2eyes_nose_2mouth_(landmarks):
    """ get [eyes, nose, mouth] landmarks from WLFW landmarks detection  
    """
    landmarks = landmarks.reshape(-1, 2)
    left_eye = np.mean(landmarks[[60, 61, 63, 64, 65, 67], :], axis=0)
    right_eye = np.mean(landmarks[[68, 69, 71, 72, 73, 75], :], axis=0)
    nose = landmarks[57]
    left_mouth = landmarks[76]
    right_mouth = landmarks[82]
    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])


def trigger_adjust_signal(image, box):
    """ check whether to trigger adjust signal through the camera moving, \n
        we use the SIFT feature point detection and matching to judge the moving of camera
    """
    global adjust_signal, last_desp, last_pts2d, last_image
    box = head2body_box1(image, box)
    if 2*(box[2]-box[0])*(box[3]-box[1]) > image.shape[0]*image.shape[1]:
        return
    pts2d, desps = filter_sift_descriptors(image, box)
    if not last_desp is None:
        camera_move_distance, matchesMask, matches = get_avg_distance(last_pts2d, last_desp, pts2d, desps)

        draw_params = dict(matchColor=(0, 255, 0),
                        singlePointColor=(255, 0, 0), 
                        matchesMask=matchesMask,
                        flags=0)
        flannmaches = cv2.drawMatchesKnn(image, pts2d, last_image, 
                            last_pts2d, matches, None, **draw_params)

        cv2.imshow('matches', flannmaches)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(1)

        # when move_distance over threshold xx, trigger the signal for adjust
        if camera_move_distance > 0.1:
            adjust_signal = True 
    last_desp  = desps
    last_pts2d = pts2d
    last_image = image


def check_adjust_signal(lms_queue, bbox_queue, desk: VirtualDesk):
    """check whether to reset the adjust signal according the lms and bbox queues,
        we reset the adjust signal when it comes such situations:
        - 1. no facebox detecting continuous ();
        - 2. current head position match desk's height and being stable;
    """
    global adjust_signal

    # situation 1
    history_boxes = bbox_queue.lastest_k(k=15)
    avg_box = np.mean(np.array(history_boxes).reshape(-1, 4), axis=0)
    if avg_box[2] < 10 and avg_box[3] < 10:
        adjust_signal = False

    # situation 2
    history_pts5p_2d = lms_queue.lastest_k(k=15)
    history_pts5p_2d = np.array(history_pts5p_2d).reshape(-1, 10)
    _, trt_vec = pose_estimate(lms_queue.smooth().reshape(-1, 2), img_size=img_size)
    # mapping the translate vector to world coordinates
    desk_height = desk.get_height()
    eye_height = trt_vec2height(trt_vec, desk_height=desk_height)

    # judge the state of head according to history_pts5p_2d
    # the position of head remaining stable indicates static, otherwise move
    diff_pts5p_2d = history_pts5p_2d[:-1, :] - history_pts5p_2d[1:, :]
    diff_pts5d_std = np.sum(np.std(diff_pts5p_2d, axis=0))
    if diff_pts5d_std > 0.5 and abs(desk_height - eye_height) < move_threshold:
        adjust_signal = False
    
    return eye_height


def adjust_actor(eye_height, desk: VirtualDesk, threshold=20):
    """adjust the height of display according to the trt_vec
    """
    desk_height = desk.get_height()
    distance = eye_height - desk_height
    if abs(distance) < threshold:
        desk.adjust(distance)


def pipeline(video_path, head_onnx_path, facelms_onnx_path):
    global adjust_signal
    head_ort_session = ort.InferenceSession(head_onnx_path)
    facelms_ort_session = ort.InferenceSession(facelms_onnx_path)

    bbox_queue = MyQueue(queue_size=30, element_dim=4,  pool_window=2)
    lms_queue  = MyQueue(queue_size=30, element_dim=10, pool_window=2)

    desk = VirtualDesk()

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    last_lms = np.ones((5, 2))
    counter = 0
    while True:
        _, ori_image = cap.read()
        ori_image = cv2.resize(ori_image, (img_size[1], img_size[0]))
        if ori_image is None:
            break

        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            head_img = ori_image[box[1]:box[3], box[0]:box[2]]
            facelms = detect_facelms_v2(head_img, facelms_ort_session)
            pts5p_2d = _2eyes_nose_2mouth_(np.array(facelms)).astype(np.int32).tolist()
            absolute_pts5p_2d = [[box[0]+x, box[1]+y] for (x, y) in pts5p_2d]

            lms_queue.enqueue(np.array(absolute_pts5p_2d).reshape(-1))
            bbox_queue.enqueue(box.reshape(-1))

            last_lms = absolute_pts5p_2d

            # for debug
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            for l in absolute_pts5p_2d:
                cv2.circle(ori_image, (l[0], l[1]), 2, (255, 0, 0), 2)
        else:
            # there should be a module that processes exceptions, e.g. no head detected,
            bbox_queue.enqueue(np.array([0,0,0,0]).reshape(-1))
            lms_queue.enqueue(np.array(last_lms).reshape(-1))

        counter = (counter + 1) % detect_interval
        if counter == 0 and not box is None:
            trigger_adjust_signal(ori_image, box)
        
        if counter == 0:
            eye_height = check_adjust_signal(lms_queue, bbox_queue, desk)

        if adjust_signal:
            print(f'need to adjust desk, current eye: {eye_height}')

        if counter == 0 and adjust_signal:
            adjust_actor(eye_height, desk)

        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    video_path = "data/WFJ_video_main2.mp4"
    head_onnx_path = "weights/lite_head_detection_simplied.onnx"
    facelms_onnx_path = "weights/facelms_112x112.onnx"
    pipeline(video_path, head_onnx_path, facelms_onnx_path)