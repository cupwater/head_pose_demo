'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-10-18 00:25:27
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort


def detect_state(input_data, ort_session, feature_dim=420):    
    input_name = ort_session.get_inputs()[0].name
    state = ort_session.run(None, {input_name: input_data})[0][0]
    landmarks = get_landmarks_from_heatmap(output, src_image.shape[:2], tgt_size=input_size)
    return landmarks, output

if __name__ == '__main__':
    onnx_path = "weights/lite_hrnet_30_coco.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    img_path = "data/test.jpg"
    image = cv2.imread(img_path)
    landmarks, _ = detect_pose(image, ort_session)

    for (x_pos,y_pos) in landmarks.tolist():
        cv2.circle(image, (x_pos,y_pos), 2, (0,0,255), -1)
    cv2.imshow('img', image)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
        exit(0)