'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-09-18 23:28:01
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort
import pdb

from utils.visualize import get_landmarks_from_heatmap

def detect_pose(ori_image, ort_session):    
    input_name = ort_session.get_inputs()[0].name

    def _preprocess(ori_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, (288, 384))
        # normalize
        input_data = (input_data/255.0 - np.array([0.485, 0.456, 0.406]))
        input_data = np.divide(input_data, np.array([0.229, 0.224, 0.225]))
        input_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        input_data = input_data.astype(np.float32)
        return input_data

    input_data = _preprocess(ori_image)
    start_time = time.time()
    output = ort_session.run(None, {input_name: input_data})[0][0]
    print("inference time:{}".format(time.time() - start_time))
    landmarks = get_landmarks_from_heatmap(output, ori_image.shape[:2])
    return landmarks

if __name__ == '__main__':
    onnx_path = "weights/lite_hrnet_30_coco.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    img_path = "data/test.jpg"
    image = cv2.imread(img_path)
    landmarks = detect_pose(image, ort_session)

    for (x_pos,y_pos) in landmarks.tolist():
        cv2.circle(image, (x_pos,y_pos), 2, (0,0,255), -1)
    cv2.imshow('img', image)
    if cv2.waitKey(-1) & 0xFF == ord('q'):
        exit(0)