'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-09-19 01:08:14
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort

import pdb

from utils.visualize import get_landmarks_from_heatmap

def detect_pose(src_image, ort_session, input_size=(384, 288)):    
    input_name = ort_session.get_inputs()[0].name

    def _preprocess(src_image, tgt_w=input_size[1], tgt_h=input_size[0]):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        src_h, src_w = input_data.shape[:2]
        # keep the ratio between width and height of source image
        if src_h*1.0/tgt_h > src_w*1.0/tgt_w:
            input_data = cv2.resize(input_data, (int(src_w*tgt_h/src_h), tgt_h))
        else:
            input_data = cv2.resize(input_data, (tgt_w, int(src_h*tgt_w/src_w)))
        # normalize
        # pdb.set_trace()
        input_data = (input_data/255.0 - np.array([0.485, 0.456, 0.406]))
        input_data = np.divide(input_data, np.array([0.229, 0.224, 0.225]))

        target_data = np.zeros((tgt_h, tgt_w, 3), dtype=np.float32)
        if src_h*1.0/tgt_h > src_w*1.0/tgt_w:
            target_data[:tgt_h, :int(src_w*tgt_h/src_h), :] = input_data
        else:
            target_data[:int(src_h*tgt_w/src_w), :tgt_w, :] = input_data
        
        target_data = np.expand_dims(np.transpose(target_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    input_data = _preprocess(src_image)
    start_time = time.time()
    output = ort_session.run(None, {input_name: input_data})[0][0]
    print("inference time:{}".format(time.time() - start_time))
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