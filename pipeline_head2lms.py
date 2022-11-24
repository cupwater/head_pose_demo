'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-11-24 15:14:43
Description: 

'''
# coding: utf8

import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_facelms_v2 import detect_facelms_v2
from utils.VirtualDesk import VirtualDesk
from utils.myqueue import MyQueue

import pdb

adjust_signal, standard_box_lms = False, None
standard_box_hw, standard_hw_ratio = [260, 190], 260.0/190
img_size, detect_interval = (540, 960), 3

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


def map_lms2world_coord(lms_queue):
    return 0


def is_standard_pose(box, lms, diff_th=0.20, ratio_hw_th=0.2, hw_th=0.2, coor_th=0.25):
    global standard_box_lms
    if standard_box_lms is None:
        standard_box_lms = (box, lms)
        return False
    lms, box = np.array(lms), np.array(box)

    eye_center, mouth_center = (lms[0] + lms[1])/2, (lms[3] + lms[4])/2 
    box_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]

    x_diff = abs(eye_center[0]-box_center[0]) + abs(mouth_center[0]-box_center[0]) + \
                    abs(lms[2][0]-box_center[0])
    x_diff = x_diff/(box[2]-box[0])
    hw_ratio = 1.0*(box[3]-box[1]) / (box[2]-box[0])

    # if the lms is nearby the center of box
    if x_diff < diff_th and abs(hw_ratio-standard_hw_ratio)<ratio_hw_th:
        # if the box width and height is similar to standard box
        if abs(1-1.0*(box[2]-box[0])/(standard_box_hw[1])) < hw_th and \
                abs(1-1.0*(box[3]-box[1])/(standard_box_hw[0])) < hw_th:
            x_bound = [(img_size[1] - standard_box_hw[1])/2, (img_size[1] + standard_box_hw[1])/2]
            # if the box is nearby the standard box in x direction
            if abs(box[0]-x_bound[0])/standard_box_hw[1]<ratio_hw_th and \
                    abs(box[2]-x_bound[1])/standard_box_hw[0]<ratio_hw_th:
                return True
    return False


def trigger_adjust_signal(standard_pose_queue, standard_th=0.88):
    """ check whether to trigger adjust signal according to the history poses \n
    """
    global adjust_signal
    avg_standard = standard_pose_queue.smooth()
    if avg_standard > standard_th:
        adjust_signal = True


def check_adjust_signal(lms_queue, bbox_queue, standard_pose_queue, \
                desk: VirtualDesk, nonstandard_th=0.2, move_threshold=0.2):
    """check whether to reset the adjust signal according the lms and bbox queues,
        we reset the adjust signal when it comes such situations:
        - 1. no facebox detecting continuous ();
        - 2. head pose is not being in standard pose continuous ();
        - 3. current head position match desk's height and being stable;
    """
    global adjust_signal
    # situation 1
    history_boxes = bbox_queue.lastest_k(k=15)
    avg_box = np.mean(np.array(history_boxes).reshape(-1, 4), axis=0)
    if avg_box[2] < 10 and avg_box[3] < 10:
        adjust_signal = False

    # situation 2
    avg_standard = standard_pose_queue.smooth()
    if avg_standard < nonstandard_th:
        adjust_signal = False

    desk_height = desk.get_height()
    if avg_standard > 0.9:
        desk_height = desk.get_height()
        # mapping the lms to world coordinate
        world_height = map_lms2world_coord(lms_queue)
        if (world_height - desk_height) < move_threshold:
            adjust_signal = False


def adjust_actor(lms_queue, desk: VirtualDesk, threshold=20):
    """adjust the height of display according to the trt_vec
    """
    desk_height = desk.get_height()
    eye_height  = lms_queue.get_average()
    distance = abs(eye_height - desk_height)
    if abs(distance) < threshold:
        desk.adjust(distance)


def pipeline(video_path, head_onnx_path, facelms_onnx_path):
    global adjust_signal, standard_box_lms
    head_ort_session = ort.InferenceSession(head_onnx_path)
    facelms_ort_session = ort.InferenceSession(facelms_onnx_path)

    bbox_queue    = MyQueue(queue_size=30, element_dim=4,  pool_window=2)
    lms_queue     = MyQueue(queue_size=30, element_dim=10, pool_window=2)
    standard_pose_queue = MyQueue(queue_size=15, element_dim=1, pool_window=2)

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

            is_standard = is_standard_pose(box, absolute_pts5p_2d)
            standard_pose_queue.enqueue(np.array([1 if is_standard else 0]))

            text = 'standard' if is_standard else 'casual' 
            cv2.putText(ori_image, text, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0,255,0), 2)
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)
            for l in absolute_pts5p_2d:
                cv2.circle(ori_image, (l[0], l[1]), 2, (255, 0, 0), 2)
        else:
            # there should be a module that processes exceptions, e.g. no head detected,
            bbox_queue.enqueue(np.array([0,0,0,0]).reshape(-1))
            lms_queue.enqueue(np.array(last_lms).reshape(-1))

        counter = (counter + 1) % detect_interval
        if counter == 0 and not box is None:
            trigger_adjust_signal(standard_pose_queue)

        if counter == 0:
            check_adjust_signal(lms_queue, bbox_queue, standard_pose_queue, desk)

        trigger = 'trigger' if adjust_signal else 'non-trigger'
        cv2.putText(ori_image, trigger, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)

        if counter == 0 and adjust_signal:
            adjust_actor(lms_queue, desk)

        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    video_path = "data/WFJ_video_main2.mp4"
    head_onnx_path = "weights/lite_head_detection_simplied.onnx"
    facelms_onnx_path = "weights/facelms_112x112.onnx"
    pipeline(video_path, head_onnx_path, facelms_onnx_path)