'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-10-18 00:33:02
Description: 

'''
# coding: utf8

import time
import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_pose import detect_pose
from utils.head2body_box import head2body_box
from utils.myqueue import MyQueue
num_points = 7


def pose_machine(pose_queue, state_queue, h_desktop):
    '''
        
    '''
    return 0


def demo(video_path, pose_onnx_path, head_onnx_path, state_onnx_path):

    pose_ort_session  = ort.InferenceSession(pose_onnx_path)
    head_ort_session  = ort.InferenceSession(head_onnx_path)
    state_ort_session = ort.InferenceSession(state_onnx_path)
    state_input_name = state_ort_session.get_inputs()[0].name

    pose_queue  = MyQueue(queue_size=30, element_dim=14)
    state_queue = MyQueue(queue_size=6,  element_dim=2)

    cap = cv2.VideoCapture(video_path)
    while True:
        _, ori_image = cap.read()
        if ori_image is None:
            break
        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            # get the bounding box for the pose detection according to head box
            body_img = head2body_box(ori_image, box)
            landmarks, _ = detect_pose(body_img, pose_ort_session)
            # update pose queue
            pose_queue.put(landmarks.reshape(-1)[:num_points*2])
            
            # get the human state
            feature = pose_queue # flat pose_queue into 1D feature
            state = state_ort_session.run(None, {state_input_name: feature})[0][0]
            # update state queue 
            state_queue.put(state)
        else:
            pose_queue.put(np.zeros(num_points*2))
            state_queue.put(np.zeros(2))
        
        # parse the state and pose, output the signal for liftable
        #  

    cap.release()
    

if __name__ == '__main__':
    video_path = "data/demo.mp4"
    head_onnx_path  = "weights/lite_head_detection_simplied.onnx"
    pose_onnx_path  = "weights/lite_hrnet_30_coco_simplied.onnx"
    state_onnx_path = "weights/pose_state_classifier.onnx"
    demo(video_path, pose_onnx_path, head_onnx_path, state_onnx_path)

    

