'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-11-28 10:22:31
Description: 

'''
# coding: utf8

import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_headposev2 import detect_head_pose, visualize_pose
# from detect_head_pose import detect_head_pose, visualize_pose
from utils.VirtualDesk import VirtualDesk
from utils.myqueue import MyQueue

adjust_signal = False
standard_box_hw, standard_hw_ratio = [195, 145], 195.0/145
img_size, detect_interval = (540, 960), 5


def map_box2world_coord(bbox_queue: MyQueue):
    smooth_pts   = bbox_queue.smooth()
    center_eye_y = smooth_pts[1] + (smooth_pts[3] - smooth_pts[1])*0.4
    eye_world_height = 0.05 * (250-center_eye_y) + 40
    return eye_world_height

def is_standard_distance(box, near_ratio=1.25, far_ratio=0.66):
    near_box_hw = [standard_box_hw[0]*near_ratio, standard_box_hw[1]*near_ratio]
    far_box_hw  = [standard_box_hw[0]*far_ratio, standard_box_hw[1]*far_ratio]
    if (box[2]-box[0]) < near_box_hw[0] and (box[2]-box[0]) > far_box_hw[0]:
        return True
    return False

def is_standard_position(box, threshold=0.5):
    box_center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    if abs(box_center[0]-img_size[1]/2)/(box[2] - box[0]) < threshold:
        return True
    return False

def is_standard_pose(headpose, yz_threshold=10, x_threshold=20):
    if headpose[0]<x_threshold and headpose[1]<yz_threshold and headpose[2]<yz_threshold:
        return True
    else:
        return False


def trigger_adjust_signal(standard_sign_queue, standard_th=0.88):
    """ check whether to trigger adjust signal according to the history poses \n
    """
    global adjust_signal
    avg_standard = standard_sign_queue.smooth()
    if avg_standard > standard_th:
        adjust_signal = True

def check_adjust_signal(bbox_queue, standard_sign_queue, \
                desk: VirtualDesk, nonstandard_th=0.2, move_threshold=2):
    """check whether to reset the adjust signal according the bbox queues and standard_sign queue,
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
    avg_standard = standard_sign_queue.smooth()
    if avg_standard < nonstandard_th:
        adjust_signal = False

    if avg_standard > 0.9:
        desk_height = desk.get_height()
        # mapping the bbox to world coordinate
        world_height = map_box2world_coord(bbox_queue)
        print(f'desktop height: {desk_height}, \t eye height: {world_height}')
        if abs(world_height - desk_height) < move_threshold:
            adjust_signal = False

def adjust_actor(bbox_queue, desk: VirtualDesk, threshold=10):
    """adjust the height of display according to the trt_vec
    """
    desk_height = desk.get_height()
    eye_height  = map_box2world_coord(bbox_queue)
    distance = eye_height - desk_height
    if abs(distance) < threshold:
        desk.adjust(distance)


def visualize(ori_image, box, pose_sign, distance_sign, position_sign, standard_sign):
    if pose_sign:
        cv2.putText(ori_image, "s-pose", (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    else:
        cv2.putText(ori_image, "c-pose", (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
    if distance_sign:
        cv2.putText(ori_image, "s-dis", (box[0], box[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    else:
        cv2.putText(ori_image, "c-dis", (box[0], box[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
    if position_sign:
        cv2.putText(ori_image, "s-posi", (box[2], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    else:
        cv2.putText(ori_image, "c-posi", (box[2], box[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
    if standard_sign:
        cv2.putText(ori_image, "standard", (box[2], box[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    else:
        cv2.putText(ori_image, "casual", (box[2], box[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
    cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)
    return ori_image


def pipeline(video_path, head_onnx_path, headpose_onnx_path):
    global adjust_signal
    head_ort_session = ort.InferenceSession(head_onnx_path)
    headpose_ort_session = ort.InferenceSession(headpose_onnx_path)

    bbox_queue    = MyQueue(queue_size=30, element_dim=4,  pool_window=2)
    standard_sign_queue = MyQueue(queue_size=15, element_dim=1, pool_window=2)

    desk = VirtualDesk()
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        _, ori_image = cap.read()
        ori_image = cv2.resize(ori_image, (img_size[1], img_size[0]))
        if ori_image is None:
            break

        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            bbox_queue.enqueue(box.reshape(-1))
            headpose = detect_head_pose(ori_image[box[1]:box[3], box[0]:box[2]], 
                                headpose_ort_session)

            image = ori_image[box[1]:box[3], box[0]:box[2]]
            image = visualize_pose(image, headpose, size=100)
            print(headpose)
            pose_sign     = is_standard_pose(headpose)
            distance_sign = is_standard_distance(box)
            position_sign = is_standard_position(box)
            standard_sign = pose_sign and distance_sign and position_sign

            standard_sign_queue.enqueue(np.array([1 if standard_sign else 0]))
            ori_image = visualize(ori_image, box, pose_sign, distance_sign, 
                            position_sign, standard_sign)
        else:
            # there should be a module that processes exceptions, e.g. no head detected,
            bbox_queue.enqueue(np.array([0,0,0,0]).reshape(-1))

        counter = (counter + 1) % detect_interval
        if counter == 0 and not box is None:
            trigger_adjust_signal(standard_sign_queue)

        if counter == 0:
            check_adjust_signal(bbox_queue, standard_sign_queue, desk)

        trigger = 'trigger' if adjust_signal else 'non-trigger'
        cv2.putText(ori_image, trigger, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        if counter == 0 and adjust_signal:
            adjust_actor(bbox_queue, desk)

        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    video_path = "data/WFJ_video_main2.mp4"
    #head_onnx_path = "weights/lite_head_detection_simplied.onnx"
    head_onnx_path = "weights/head_detection_RFB_slim_320x240.onnx"
    # headpose_onnx_path = "weights/head_pose.onnx"
    headpose_onnx_path = "weights/headpose_mbnv2.onnx"
    pipeline(video_path, head_onnx_path, headpose_onnx_path)