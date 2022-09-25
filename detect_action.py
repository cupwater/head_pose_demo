'''
Author: Peng Bo
Date: 2022-09-16 21:43:34
LastEditTime: 2022-09-25 17:13:26
Description: 

'''
import cv2
import pdb
import numpy as np
import onnxruntime as ort

action_list = [
    '吃零食',
    '喝水',
    '咬手指',
    '瞌睡',
    '发呆',
    '玩橡皮',
    '离开书桌',
    '正常'
]

class MyQueue:
    def __init__(self, queue_size=12, element_dim=len(action_list)):
        self.queue_size  = queue_size
        self.queue = np.zeros((queue_size, element_dim), dtype=np.float32)
        self.head = -1
        self.tail = -1

    def get_average(self):
        if self.element_num == self.queue_size:
            return np.mean(self.queue, axis=1)
        else:
            if self.head >= self.tail:
                return np.mean( self.queue[self.tail:(self.head+1)], axis=0 )
            else:
                temp = np.concatenate([ self.queue[self.tail:], self.queue[:self.head] ], axis=0) 
                return np.mean(temp, axis=0)

    def put(self, element):
        if self.head != -1:
            self.queue[(self.head+1)%self.queue_size] = element
            self.head = (self.head+1) % self.queue_size
            if self.head == self.tail:
                self.get()
        else:
            self.head = self.tail = 0
            self.queue[self.head] = element[0]
    
    def get(self):
        if self.head == -1:
            print("error, the queue is empty")
            return None
        else:
            if self.tail == self.head:
                self.head = self.tail = -1
                return self.queue[0]
            self.tail = (self.tail+1) % self.queue_size
    
    def element_num(self):
        if (self.head + 1) % self.queue_size == self.tail:
            return self.queue_size
        elif self.head == self.tail:
            return 1
        else:
            return self.head - self.tail


def detect_action(video, smodel, tmodel, fmodel=None, dets_res=None, step=4, frame_len=12, input_size=(320, 240)):
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
        image = (image - np.array([127, 127, 127])) / 128.0
        image = cv2.resize(image, input_size)
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        return image

    squeue  = MyQueue(queue_size=frame_len)
    tqueue  = MyQueue(queue_size=frame_len)
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
                squeue.put(np.exp(x)/sum(np.exp(x)))
                x = np.ones( len(action_list) )
                tqueue.put(np.exp(x)/sum(np.exp(x)))
            else:
                processed_frame = _preprocess(frame)
                # extract features and recogntiion using temporal segment network
                rgbdiff = processed_frame - _preprocess(previous_frame)
                squeue.put(smodel.run(None, {sname: processed_frame})[0])
                tqueue.put(tmodel.run(None, {tname: rgbdiff})[0])
            # Consensus the K frame as the final results
            predict  = squeue.get_average() + tqueue.get_average()
            idx = np.argmax(predict)
            prob = predict[idx]
            semantic_label = action_list[idx]
            inference_res.append((semantic_label, prob))
        frame_idx += 1
    return inference_res

if __name__ == '__main__':
    spatial_ort_session  = ort.InferenceSession("weights/r18_smodel_simplied.onnx")
    temporal_ort_session = ort.InferenceSession("weights/r34_tmodel_simplied.onnx")
    # fusion_ort_session   = ort.InferenceSession("weights/lite_head_detection_simplied.onnx")
    video_path = "data/demo.mp4"
    results = detect_action(video_path, spatial_ort_session, temporal_ort_session)