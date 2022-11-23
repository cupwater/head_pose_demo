'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-11-14 18:31:55
Description: 

'''
# coding: utf8

import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_head_pose import detect_head_pose, visualize_pose
from utils.myqueue import MyQueue
from utils.head2body_box import head2box

num_points = 7
states_list = ["动作", "静止"]

class VirtualDesk:
    def __init__(self, init_height=120, height_range=(100, 160)):
        self.height_range  = height_range
        self.height = init_height
    
    def get_height(self):
        return self.height
    
    def up(self, distance):
        if self.height+distance >= self.height_range[1]:
            print("exceed max height, will keep max height")
            self.height = self.height_range[1]
        else:
            self.height += distance

    def down(self, distance):
        if self.height-distance <= self.height_range[0]:
            print("exceed min height, will keep min height")
            self.height = self.height_range[0]
        else:
            self.height -= distance
    
    def adjust(self, distance):
        if distance<0:
            self.down(abs(distance))
        else:
            self.up(abs(distance))

def pose2deskHeight(pose, scale=1.0, bias=1.0):
    """
        judge whether pose matches the height of desk.
        Since the relation between pose height and the target height should be lineared,
            we just need to multiply the pose by scale and plus bias to get the target height.
        scale: need to setting when initialize
        bias: need to setting when initialize
        return 0 if match, else the moving distance the desk need to adjust
    """
    # First, need to calibrate the standard pose and desk height, 
    # and get the mapping function between pose and desk height 
    pose = pose.reshape(-1, 2)
    return scale*np.max(pose[1, :]) + bias

def pose_machine(headpose_queue, state_queue, desk, threshold=5):
    """
        liftable desk driver according to pose&state queues and desk_height
    """
    current_state  = states_list[np.argmax(state_queue.get_average())]
    # print(current_state)
    if current_state == "运动":
        return
    # get current pose and state
    current_pose   = headpose_queue.get_average()
    distance = pose2deskHeight(current_pose) - desk.get_height()
    # only when distance exceed threshold, the desk need to adjust
    if abs(distance) > threshold:
        desk.adjust(distance)
         
def pipeline(video_path, headpose_onnx_path, head_onnx_path, state_onnx_path):
    
    headpose_ort_session = ort.InferenceSession(headpose_onnx_path)
    head_ort_session = ort.InferenceSession(head_onnx_path)
    state_ort_session = ort.InferenceSession(state_onnx_path)
    state_input_name = state_ort_session.get_inputs()[0].name

    head_position_queue = MyQueue(queue_size=60, element_dim=4, pool_window=1)
    state_queue = MyQueue(queue_size=6,  element_dim=2)
    desk = VirtualDesk()

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture(0)
    while True:
        _, ori_image = cap.read()
        if ori_image is None:
            break
        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            correct_box = head2box(ori_image, box)
            head_img = ori_image[correct_box[1]:correct_box[3], correct_box[0]:correct_box[2]]
            headpose = detect_head_pose(head_img, headpose_ort_session)
            head_position_queue.enqueue(np.array(box))

            # get the human state and update state queue
            feature = np.array(head_position_queue.to_feature()).astype(
                np.float32).reshape(1, -1)
            # state = state_ort_session.run(
            #     None, {state_input_name: feature})[0][0]
            state = [0, 1.0]
            # smooth the state or not ?
            state_queue.enqueue(state)
            visualize_pose(ori_image, headpose, size=100)
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        else:
            # there should be a module that processes exceptions, e.g. no head detected, 
            head_position_queue.enqueue(np.zeros(4))
            state_queue.enqueue(np.array([0, 1.0], dtype=np.float32))

        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        pose_machine(head_position_queue, state_queue, desk)
    cap.release()

if __name__ == '__main__':
    video_path = "data/WFJ_video_main2.mp4"
    head_onnx_path = "weights/lite_head_detection_simplied.onnx"
    head_pose_onnx_path = "weights/head_pose.onnx"
    state_onnx_path = "weights/pose_state_classifier.onnx"
    pipeline(video_path, head_pose_onnx_path, head_onnx_path, state_onnx_path)