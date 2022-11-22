# coding: utf8
'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-11-22 10:26:31
Description: 

'''


import cv2
import numpy as np
import onnxruntime as ort

from detect_face_lms import detect_face_lms, img_size
from utils.solve_pose import pose_estimate
from utils.myqueue import MyQueue

import pdb

num_points = 7
states_list = ["动作", "静止"]
class VirtualDesk:
    def __init__(self, init_height=120, height_range=(50, 160)):
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


def pipeline(video_path, face_lms_path, state_onnx_path):
    facelms_ort_session = ort.InferenceSession(face_lms_path)
    state_ort_session = ort.InferenceSession(state_onnx_path)
    state_input_name = state_ort_session.get_inputs()[0].name
    bbox_queue  = MyQueue(queue_size=60, element_dim=4,  pool_window=4)
    lms_queue   = MyQueue(queue_size=60, element_dim=10, pool_window=4)
    state_queue = MyQueue(queue_size=6,  element_dim=2)
    desk = VirtualDesk()

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    while True:
        _, ori_image = cap.read()
        image = cv2.resize(ori_image, (img_size[1], img_size[0]))
        if ori_image is None:
            break
        # only return the most reliable detection result
        dets, landms = detect_face_lms(image, facelms_ort_session, input_size=(img_size[1], img_size[0]))
        if len(dets) > 0:
            box = list(map(int, dets[0][:4]))
            lms = landms.astype(np.int32).reshape(-1, 2)

            bbox_queue.enqueue(np.array(box))
            lms_queue.enqueue(lms.reshape(-1))

            rot_vec, trt_vec = pose_estimate(lms_queue.smooth().reshape(-1, 2))
            print('-------------------------\n', rot_vec, '||||\n', trt_vec,
              '\n-------------------------\n')
            # get the human state and update state queue
            feature = np.array(bbox_queue.to_feature()).astype(
                np.float32).reshape(1, -1)
            # state = state_ort_session.run(None, {state_input_name: feature})[0][0]
            state = [0, 1.0]
            # smooth the state or not ?
            state_queue.enqueue(state)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            for l in lms.tolist():
                cv2.circle(image, (l[0], l[1]), 1, (0, 0, 255), 4)
        else:
            # there should be a module that processes exceptions, e.g. no head detected, 
            bbox_queue.enqueue(np.zeros(4))
            state_queue.enqueue(np.array([0, 1.0], dtype=np.float32))
        cv2.imshow('annotated', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    video_path = "data/demo.mp4"
    face_lms_path = "weights/mbnv3_640x360.onnx"
    state_onnx_path = "weights/pose_state_classifier.onnx"
    pipeline(video_path, face_lms_path, state_onnx_path)