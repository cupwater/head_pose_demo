'''
Author: Peng Bo
Date: 2022-09-18 10:56:03
LastEditTime: 2022-09-18 23:36:22
Description: 

'''
# coding: utf8

import time
import cv2
import onnxruntime as ort

from detect_head import detect_head
from detect_pose import detect_pose
from utils.head2body_box import head2body_box


def demo(video_path, pose_onnx_path, head_onnx_path):

    pose_ort_session = ort.InferenceSession(pose_onnx_path)
    head_ort_session = ort.InferenceSession(head_onnx_path)

    cap = cv2.VideoCapture(video_path)
    while True:
        _, ori_image = cap.read()
        # ori_image = cv2.imread('data/test.jpg')
        if ori_image is None:
            break
        box = detect_head(ori_image, head_ort_session)
        if not box is None:
            # get the bounding box for the pose detection according to head box
            body_img = head2body_box(ori_image, box)
            landmarks = detect_pose(body_img, pose_ort_session)

            # visualize the detecting results
            for (x_pos,y_pos) in landmarks.tolist():
                cv2.circle(body_img, (x_pos,y_pos), 4, (0,0,255), -1)
            cv2.rectangle(ori_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            # ori_image = cv2.resize(ori_image, (0, 0), fx=0.7, fy=0.7)

            # cv2.imshow('body', body_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

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
