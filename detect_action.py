'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-11-24 18:14:07
Description: 

'''
import cv2
import pdb
import numpy as np
import onnxruntime as ort
import os

from utils.myqueue import MyQueue

action_list = [
    'normal',
    'chew',
    'drink',
    'eating',
    'leaving',
    'playing'
]

def detect_action(video, smodel, tmodel, step=4, frame_len=12, input_size=(224, 224)):
    '''
        smodel: spatial model to extract spatial feature.
        tmodel: temporal model to extract temporal feature.
        step: sample interval.
        frame_len: the length of frames used for recognition.
        input_size: the input size for feature extraction model.
    '''
    cap = cv2.VideoCapture(video)
    sname = smodel.get_inputs()[0].name
    tname = tmodel.get_inputs()[0].name

    # pre-process the input image 
    def _preprocess(ori_image):
        h,w = ori_image.shape[:2]
        if h>w:
            new_h, new_w = int(224*h/w), 224
            image = cv2.resize(ori_image, (new_w, new_h))
            start_idx = int((new_h-224)/2)
            image = image[start_idx:(start_idx+224), :]
        else:
            new_h, new_w = 224, int(224*w/h)
            image = cv2.resize(ori_image, (new_w, new_h))
            start_idx = int((new_w-224)/2)
            image = image[:, start_idx:(start_idx+224), :]
        cv2.imshow('resize', image)
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
        image = (image - np.array([123.675, 116.28, 103.53])) 
        image = np.divide(image, np.array([58.395, 57.12, 57.375]))
        # keep the hw ratio, and crop the center area
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        return image

    squeue  = MyQueue(queue_size=frame_len, element_dim=len(action_list))
    tqueue  = MyQueue(queue_size=frame_len, element_dim=len(action_list))
    inference_res = []
    frame_idx = 1
    while(True):
        _, frame = cap.read()
        if frame is None:
            break
        # used to generate rgbdiff
        if (frame_idx+1) % step == 0:
            previous_frame = frame
        elif frame_idx % step == 0:

            processed_frame = _preprocess(frame)
            # extract features and recogntiion using temporal segment network
            rgbdiff = processed_frame - _preprocess(previous_frame)
            squeue.enqueue(smodel.run(None, {sname: processed_frame})[0])
            tqueue.enqueue(tmodel.run(None, {tname: rgbdiff})[0])

            # Consensus the K frame as the final results
            predict  = (squeue.get_average() + tqueue.get_average()) / 2
            predict = np.exp(predict)/np.exp(predict).sum()
            idx = np.argmax( predict )
            prob = predict[idx]
            semantic_label = action_list[idx]
            inference_res.append((idx, semantic_label, prob))
        frame_idx += 1

        cv2.imshow('annotated', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return inference_res

if __name__ == '__main__':
    spatial_ort_session  = ort.InferenceSession("weights/test_simplify.onnx")
    temporal_ort_session = ort.InferenceSession("weights/test_simplify.onnx")
    videos_list = open('data/val.lst').readlines()
    predict_labels, labels = [], []
    prefix = 'data/val_videos'
    for line in videos_list: 
        video_path, label = line.strip().split(' ')
        labels.append(int(label))
        result = detect_action(os.path.join(prefix, video_path), spatial_ort_session, temporal_ort_session)
        idx_list = [v[0] for v in result]
        predict_labels.append(idx_list)
