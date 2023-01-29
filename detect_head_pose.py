'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2022-11-14 10:29:11
Description: 

'''
import time
import cv2
import numpy as np
import onnxruntime as ort
import pdb

# def visualize_head_pose():
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

    roll, yaw, pitch = headpose[2], headpose[0], headpose[1]
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

    input_data = _preprocess(src_image)
    start_time = time.time()
    angles = ort_session.run(None, {input_name: input_data})
    # print("inference time:{}".format(time.time() - start_time))
    headpose = [angles[0].item(), angles[1].item(), angles[2].item()]
    headpose = [abs(headpose[0]), abs(headpose[1]), abs(headpose[2])]
    return headpose

if __name__ == '__main__':
    onnx_path = "weights/head_pose.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # img_path = "data/heads/2.jpg"
        # image = cv2.imread(img_path)
        headpose = detect_head_pose(frame, ort_session)
        image = visualize_pose(frame, headpose, size=100)
        cv2.imshow("Result", image)

        key = cv2.waitKey(1)
        if key==27 or key == ord("q"):
            exit(0)