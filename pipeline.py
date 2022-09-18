'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-09-18 11:16:57
Description: 

'''
# coding: utf8

import time
import cv2
import numpy as np
import onnxruntime as ort

from detect_head import detect_head
from detect_pose import detect_pose

def demo(video_path, pose_onnx_path, head_onnx_path):

    pose_ort_session = ort.InferenceSession(pose_onnx_path)
    head_ort_session = ort.InferenceSession(head_onnx_path)

    cap = cv2.VideoCapture(video_path)
    while True:
        _, ori_image = cap.read()
        if ori_image is None:
            break
        boxes, _, _ = detect_head(ori_image, head_ort_session)
        # visualize the detecting results
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        ori_image = cv2.resize(ori_image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('annotated', ori_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = "data/demo.mp4"
    head_onnx_path = "weights/lite_head_detection.onnx"
    pose_onnx_path = "weights/lite_hrnet_30_coco.onnx"
    demo(video_path, pose_onnx_path, head_onnx_path)
