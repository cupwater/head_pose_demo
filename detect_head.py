# coding: utf8
'''
Author: Peng Bo
Date: 2022-09-18 10:17:57
LastEditTime: 2022-09-18 23:33:45
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort

from utils.nms import hard_nms


def predict(imgsize, confidences, boxes, prob_threshold=0.6, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k,)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= imgsize[0]
    picked_box_probs[:, 1] *= imgsize[1]
    picked_box_probs[:, 2] *= imgsize[0]
    picked_box_probs[:, 3] *= imgsize[1]
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def detect_head(ori_image, ort_session):
    h, w, _ = ori_image.shape
    input_name = ort_session.get_inputs()[0].name

    def _preprocess(ori_image):
        # pre-process the input image 
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image = (image - np.array([127, 127, 127])) / 128.0
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        return image

    image = _preprocess(ori_image)
    confidences, boxes = ort_session.run(None, {input_name: image})
    boxes, labels, probs = predict((w,h), confidences, boxes, prob_threshold=0.6)

    if len(boxes) == 0:
        return None
    # get the max area head and return
    max_area, max_idx = -1, -1
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        cur_area = abs((box[2] - box[0]) * (box[3] - box[1])) 
        if max_area < cur_area:
            max_area = cur_area
            max_idx = i
    return boxes[max_idx]



if __name__ == '__main__':
    img_path = "data/test.jpg"
    image = cv2.imread(img_path)

    onnx_path = "weights/lite_head_detection.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    box = detect_head(image, ort_session)

    # visualize the detecting results
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow('annotated', image)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
        exit(0)