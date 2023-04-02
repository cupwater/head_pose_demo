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
    'node',
    'normal',
    'eating',
    'daze',
    'leaving',
    'playing',
    'bit_finger',
    'drinking',
]


def detect_action(video, smodel, tmodel, step=4, frame_len=12, input_size=224):
    '''
        视频动作识别函数，采用 two-stream 方法，即利用smodel 和 tmodel 两个模型分别识别 spatial stream 和 motion stream，然后再对识别结果进行融合得到当前的识别结果。
        spatial stream，即每 step 帧选取一帧共选取 frame_len 帧作为一个视频片段并对其进行动作识别，作为spatial stream 的识别结果。
        motion  stream，即每 step 帧选取两相邻帧并用帧间之差，提供选取 frame_len 张帧间之差作为一个视频片段并对其进行动作识别，作为motion stream 的识别结果。
        由于端上部署目前并不支持 Conv3D 等temporal操作，这里我们采用 TSN 算法，即视频片段的识别是通过对每一帧进行识别然后进行融合，作为视频片段的动作识别结果。

        smodel: spatial model
        tmodel: temporal(motion) model 
        step: 采样间隔
        frame_len: 每次识别的视频片段包含的帧数
        input_size: smodel/tmodel 模型的输入大小
    '''
    cap = cv2.VideoCapture(video)

    # 获取 smodel 和 tmodel onnx模型输入的名字
    sname, tname = smodel.get_inputs()[0].name, tmodel.get_inputs()[0].name
    # 对视频帧进行预处理函数（包括归一化、类型变换、将HxWxC 通道顺序变成 CxHxW 顺序等操作）

    def _preprocess(ori_image):
        h, w = ori_image.shape[:2]
        if h > w:
            new_h, new_w = int(input_size*h/w), input_size
            image = cv2.resize(ori_image, (new_w, new_h))
            start_idx = int((new_h-input_size)/2)
            image = image[start_idx:(start_idx+input_size), :]
        else:
            new_h, new_w = input_size, int(input_size*w/h)
            image = cv2.resize(ori_image, (new_w, new_h))
            start_idx = int((new_w-input_size)/2)
            image = image[:, start_idx:(start_idx+input_size), :]
        cv2.imshow('resize', image)
        image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
        image = (image - np.array([123.675, 116.28, 103.53]))
        image = np.divide(image, np.array([58.395, 57.12, 57.375]))
        # keep the hw ratio, and crop the center area
        image = np.expand_dims(np.transpose(image, [2, 0, 1]), axis=0)
        image = image.astype(np.float32)
        return image

    # squeue 是用于保存 smodel 对每一帧识别结果的队列
    squeue = MyQueue(queue_size=frame_len, element_dim=len(action_list))
    # tqueue 是用于保存 tmodel 对每一帧识别结果的队列
    tqueue = MyQueue(queue_size=frame_len, element_dim=len(action_list))
    # 用于保存当前视频的动作识别结果
    recognition_res = []
    # 由于我们不会识别视频每一帧，而是每个 step 帧识别一次，
    # 所以这里通过 frame_idx % step == 0 用来判断是否进行识别
    frame_idx = 1
    while (True):
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
            #print(tmodel.run(None, {sname: processed_frame})[0].shape)
            squeue.enqueue(smodel.run(None, {sname: processed_frame})[0])
            tqueue.enqueue(tmodel.run(None, {tname: rgbdiff})[0])

            # Consensus the K frame as the final results
            # 将 smodel 和 tmodel 的识别结果进行简单的融合（平均融合）
            predict = (squeue.get_average() + tqueue.get_average()) / 2
            predict = np.exp(predict)/np.exp(predict).sum()
            #print(predict.shape)
            idx = np.argmax(predict)
            prob = predict[idx]
            semantic_label = action_list[idx]
            # 将最终的识别结果保存到列表中
            recognition_res.append((idx, semantic_label, prob))
        frame_idx += 1
    cap.release()
    return recognition_res


if __name__ == '__main__':
    spatial_ort_session = ort.InferenceSession(
        "weights/action_demo/tsn_mobilenetv2_1x1x8_100e_lege8actions_rgb.onnx")
    temporal_ort_session = ort.InferenceSession(
        "weights/action_demo/tsn_mobilenetv2_1x1x8_100e_lege8actions_rgbdiff.onnx")
    videos_list = open('data/val_list.txt').readlines()
    predict_labels, labels = [], []
    prefix = 'data/val_videos/'
    for line in videos_list:
        print(f"process {line.strip()}")
        video_path = line.strip()
        label = action_list.index(video_path.split('/')[0])
        labels.append(int(label))
        result = detect_action(os.path.join(
            prefix, video_path), spatial_ort_session, temporal_ort_session)
        idx_list = [v[0] for v in result]
        predict_labels.append(max(idx_list, key=idx_list.count))
    pdb.set_trace()
