'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-09-18 11:16:33
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort

from utils.visualize import get_landmarks_from_heatmap, visualize_heatmap

def detect_pose(ori_image, ort_session):
    # ort_session = ort.InferenceSession(onnx_path)
    # ori_image = cv2.imread(img_path)
    
    input_name = ort_session.get_inputs()[0].name

    def _preprocess(ori_image):
        # pre-process the input image 
        image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (288, 384))
        # normalize
        image = (image/255.0 - np.array([0.485, 0.456, 0.406]))
        image = np.divide(image, np.array([0.229, 0.224, 0.225]))

        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        return image

    image = _preprocess(ori_image)
    start_time = time.time()
    output = ort_session.run(None, {input_name: image})[0][0]
    print("inference time:{}".format(time.time() - start_time))

    landmarks = get_landmarks_from_heatmap(output)
    # img = visualize_heatmap(image, landmarks)
    
    return landmarks
    # cv2.imshow('img', img)
    # if cv2.waitKey(-1) & 0xFF == ord('q'):
    #     exit(0)

if __name__ == '__main__':
    img_path = "data/test.jpg"
    image = cv2.imread(img_path)

    onnx_path = "weights/lite_hrnet_30_coco.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    
    detect_pose(image, ort_session)