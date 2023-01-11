'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2023-01-11 16:20:20
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort

import pdb
import torch

from models.mobilenetv2 import mobilenet_v2


def visualize_pose(img, headpose, size=100):
    
    def _EulerToMatrix(roll, yaw, pitch):
        import math
        roll = roll / 180 * np.pi
        yaw = yaw / 180 * np.pi
        pitch = pitch / 180 * np.pi
        Rz = [[math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1]]
        Ry = [[math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]]
        Rx = [[1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]]
        matrix = np.matmul(Rx, Ry)
        matrix = np.matmul(matrix, Rz)
        return matrix

    roll, yaw, pitch = headpose[0], headpose[1], headpose[2]
    tdx, tdy = img.shape[1]/2, img.shape[0]/2
    matrix = _EulerToMatrix(-roll, -yaw, -pitch)

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size
    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0]+tdx), int(-Xaxis[1]+tdy)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(-Yaxis[0]+tdx), int(Yaxis[1]+tdy)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0]+tdx), int(-Zaxis[1]+tdy)), (255, 0, 0), 2)
    return img

def np_softmax(x):
    return(np.exp(x)/np.exp(x).sum())

# pre-process the image 
def preprocess(src_image, input_size=(224, 244)):
    input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    input_data = cv2.resize(input_data, input_size)
    input_data = (input_data/255.0 - np.array([0.485, 0.456, 0.406]))
    input_data = np.divide(input_data, np.array([0.229, 0.224, 0.225]))
    target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
    target_data = target_data.astype(np.float32)
    return target_data

# convert the model output to the angles 
def postprocess(yaw_vec, pitch_vec, roll_vec):
    idx_array = [idx for idx in range(66)]
    yaw   = np.sum(np_softmax(yaw_vec)*idx_array, axis=1)*3 - 99
    pitch = np.sum(np_softmax(pitch_vec)*idx_array, axis=1)*3 - 99
    roll  = np.sum(np_softmax(roll_vec)*idx_array, axis=1)*3 - 99
    return [yaw, pitch, roll]

# use onnx model to detect head pose
def detect_head_pose(src_image, ort_session, input_size=(224, 224)):    
    input_name = ort_session.get_inputs()[0].name
    input_data = preprocess(src_image)
    yaw_vec, pitch_vec, roll_vec = ort_session.run(None, {input_name: input_data})
    yaw, pitch, roll = postprocess(yaw_vec, pitch_vec, roll_vec)
    return yaw, pitch, roll

# using pytorch model to detect head pose
def detect_headpose_pt(src_image, model, input_size=(224, 224)):
    input_data = torch.from_numpy(preprocess(src_image))
    yaw_vec, pitch_vec, roll_vec= model(input_data)
    yaw_vec, pitch_vec, roll_vec = yaw_vec.detach().numpy(), pitch_vec.detach().numpy(), roll_vec.detach().numpy()
    yaw, pitch, roll = postprocess(yaw_vec, pitch_vec, roll_vec)
    return yaw, pitch, roll

if __name__ == '__main__':
    ort_session = ort.InferenceSession("weights/mobilenetv2/mobilenetv2.onnx")
    model = mobilenet_v2(num_classes=66)
    model.load_state_dict(torch.load('weights/mobilenetv2/mobilenetv2.pt', map_location='cpu'))

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            break
        headpose = detect_head_pose(image, ort_session)
        headpose_pt = detect_headpose_pt(image, model)
        pdb.set_trace()
        image = visualize_pose(image, headpose, size=100)
        cv2.imshow("Result", image)

        key = cv2.waitKey(1)
        if key==27 or key == ord("q"):
            exit(0)