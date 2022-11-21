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
from detect_facelms_v2 import detect_facelms_v2
from utils.solve_pose import pose_estimate
from utils.myqueue import MyQueue

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
        

def pipeline(video_path, head_onnx_path, facelms_onnx_path, state_onnx_path):
    
    head_ort_session = ort.InferenceSession(head_onnx_path)
    facelms_ort_session = ort.InferenceSession(facelms_onnx_path)
    state_ort_session = ort.InferenceSession(state_onnx_path)

    state_input_name = state_ort_session.get_inputs()[0].name
    head_position_queue = MyQueue(queue_size=60, element_dim=4, pool_window=1)
    state_queue = MyQueue(queue_size=6,  element_dim=2)
    desk = VirtualDesk()


    def _2eyes_nose_2mouth_(landmarks):
        left_eye  = np.mean(landmarks[60:67, :], axis=0)
        right_eye = np.mean(landmarks[68:75, :], axis=0) 
        nose      = landmarks[54]
        left_mouth = landmarks[76]
        right_mouth = landmarks[82]
        return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])

    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    while True:
        _, ori_image = cap.read()
        if ori_image is None:
            break
        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            head_img = ori_image[box[1]:box[3], box[0]:box[2]]
            facelms = detect_facelms_v2(head_img, facelms_ort_session)
            facelms = [ [box[0]+x, box[1]+y] for (x, y) in facelms.astype(np.int32).tolist()]

            pts5p_2d = _2eyes_nose_2mouth_(np.array(facelms))
            rot_vec, trt_vec = pose_estimate(pts5p_2d, img_size=ori_image.shape)
            print('-------------------------\n', rot_vec, '||||\n', trt_vec,
              '\n-------------------------\n')
            #head_position_queue.enqueue(np.array(box))

            ## get the human state and update state queue
            #feature = np.array(head_position_queue.to_feature()).astype(
            #    np.float32).reshape(1, -1)
            ## state = state_ort_session.run(
            ##     None, {state_input_name: feature})[0][0]
            #state = [0, 1.0]
            ## smooth the state or not ?
            #state_queue.enqueue(state)
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            for l in facelms:
                cv2.circle(ori_image, (l[0], l[1]), 2, (255,0,0), 2)
        #else:
        #    # there should be a module that processes exceptions, e.g. no head detected, 
        #    head_position_queue.enqueue(np.zeros(4))
        #    state_queue.enqueue(np.array([0, 1.0], dtype=np.float32))

        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(-1) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == '__main__':
    video_path = "data/WFJ_video_main2.mp4"
    head_onnx_path = "weights/lite_head_detection_simplied.onnx"
    facelms_onnx_path = "weights/facelms_112x112.onnx"
    state_onnx_path = "weights/pose_state_classifier.onnx"
    pipeline(video_path, head_onnx_path, facelms_onnx_path, state_onnx_path)