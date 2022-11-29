# coding: utf8
'''
Author: Peng Bo
Date: 2022-09-18 10:17:57
LastEditTime: 2022-09-25 10:56:04
Description: 

'''

import cv2
import numpy as np
import onnxruntime as ort


def detect_facelms_v2(ori_image, ort_session, input_size=(112, 112)):
    h, w, _ = ori_image.shape
    input_name = ort_session.get_inputs()[0].name
    def _preprocess(ori_image):
        # pre-process the input image 
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_size)
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        image = image / 255.0
        return image
    image = _preprocess(ori_image)
    landmarks = ort_session.run(None, {input_name: image})[0]
    landmarks = landmarks.reshape(-1, 2)* np.array([w, h])
    return landmarks

if __name__ == '__main__':
    img_path = "data/test_facelms.jpg"
    image = cv2.imread(img_path)
    onnx_path = "weights/facelms_112x112.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    lms = detect_facelms_v2(image, ort_session)
    for l in lms.astype(np.int32).tolist():
        cv2.circle(image, (l[0], l[1]), 2, (255,0,0), 2)
    # visualize the detecting results
    cv2.imshow('annotated', image)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
        exit(0)