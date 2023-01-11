'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2023-01-11 16:06:51
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort

import pdb

idx_array = [idx for idx in range(66)]

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


def detect_head_pose(src_image, ort_session, input_size=(224, 224)):    
    input_name = ort_session.get_inputs()[0].name
    def _preprocess(src_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, input_size)
        input_data = (input_data/255.0 - np.array([0.485, 0.456, 0.406]))
        input_data = np.divide(input_data, np.array([0.229, 0.224, 0.225]))
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    def np_softmax(x):
        return(np.exp(x)/np.exp(x).sum())

    input_data = _preprocess(src_image)
    yaw, pitch, roll = ort_session.run(None, {input_name: input_data})
    # print("inference time:{}".format(time.time() - start_time))
    yaw_pred   = np.sum(np_softmax(yaw)*idx_array, axis=1)*3 - 99
    pitch_pred = np.sum(np_softmax(pitch)*idx_array, axis=1)*3 - 99
    roll_pred  = np.sum(np_softmax(roll)*idx_array, axis=1)*3 - 99
    return [yaw_pred, pitch_pred, roll_pred]


def detect_headpose_pt(src_image, model, input_size=(224, 224)):
    def np_softmax(x):
        return(np.exp(x)/np.exp(x).sum())

    def _preprocess(src_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, input_size)
        input_data = (input_data/255.0 - np.array([0.485, 0.456, 0.406]))
        input_data = np.divide(input_data, np.array([0.229, 0.224, 0.225]))
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    import torch
    input_data = torch.from_numpy(_preprocess(src_image))

    yaw, pitch, roll = model(input_data)
    yaw, pitch, roll = yaw.detach().numpy(), pitch.detach().numpy(), roll.detach().numpy()
    yaw_pred   = np.sum(np_softmax(yaw)*idx_array, axis=1)*3 - 99
    pitch_pred = np.sum(np_softmax(pitch)*idx_array, axis=1)*3 - 99
    roll_pred  = np.sum(np_softmax(roll)*idx_array, axis=1)*3 - 99
    return [yaw_pred, pitch_pred, roll_pred]


if __name__ == '__main__':
    onnx_path = "weights/mobilenetv2/mobilenetv2.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    cap = cv2.VideoCapture(0)
    import torch
    from models.mobilenetv2 import mobilenet_v2
    model = mobilenet_v2(num_classes=66)
    model.load_state_dict(torch.load('weights/mobilenetv2/mobilenetv2.pt', map_location='cpu'))
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