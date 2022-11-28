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

from utils.myqueue import MyQueue

action_list = [
    'normal',
    'chew',
    'drink',
    'eating',
    'leaving',
    'playing'
]

def detect_action(video, smodel, tmodel, fmodel=None, dets_res=None, step=4, frame_len=12, input_size=(224, 224)):
    '''
        smodel: spatial model to extract spatial feature.
        tmodel: temporal model to extract temporal feature.
        fmodel: fusion model to fusing spatial and temporal feature for final recognition.
        dets_res: person detection results for video.
        step: sample interval.
        frame_len: the length of frames used for recognition.
        input_size: the input size for feature extraction model.
    '''
    cap = cv2.VideoCapture(video)
    sname = smodel.get_inputs()[0].name
    tname = tmodel.get_inputs()[0].name

    # pre-process the input image 
    def _preprocess(ori_image):
        image = cv2.cvtColor(ori_image.astype(np.float32), cv2.COLOR_BGR2RGB)
        image = (image - np.array([123.675, 116.28, 103.53])) 
        image = np.divide(image, np.array([58.395, 57.12, 57.375]))
        image = cv2.resize(image, input_size)
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
            if not dets_res is None and dets_res[frame_idx] is None:
                x = np.ones( len(action_list) )
                squeue.enqueue(np.exp(x)/sum(np.exp(x)))
                x = np.ones( len(action_list) )
                tqueue.enqueue(np.exp(x)/sum(np.exp(x)))
            else:
                processed_frame = _preprocess(frame)
                # extract features and recogntiion using temporal segment network
                rgbdiff = processed_frame - _preprocess(previous_frame)
                squeue.enqueue(smodel.run(None, {sname: processed_frame})[0])
                tqueue.enqueue(tmodel.run(None, {tname: rgbdiff})[0])
            # Consensus the K frame as the final results
            predict  = (squeue.get_average() + tqueue.get_average()) / 2
            idx = np.argmax(predict)
            prob = predict[idx]
            semantic_label = action_list[idx]
            inference_res.append((semantic_label, prob))
        frame_idx += 1
    return inference_res

if __name__ == '__main__':
    spatial_ort_session  = ort.InferenceSession("weights/test_simplify.onnx")
    temporal_ort_session = ort.InferenceSession("weights/test_simplify.onnx")
    video_path = "data/demo.mp4"
    results = detect_action(video_path, spatial_ort_session, temporal_ort_session)
    print(results)