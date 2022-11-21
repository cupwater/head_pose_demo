'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2022-11-21 21:01:12
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort
import pdb

from utils.box_util import decode, decode_landm, gen_bbox
from utils.solve_pose import pose_estimate
from utils.nms import py_cpu_nms


img_size = (360, 640)
anchors, variance = gen_bbox(img_size)
scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])

def detect_face_lms(src_image, ort_session, input_size=(320, 180), conf_thre=0.6, nms_thre=00.5):
    input_name = ort_session.get_inputs()[0].name
    def _preprocess(src_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, input_size)
        input_data = (input_data - np.array([104, 117, 123]))
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data
    
    def _post_process(loc, conf, landms):
        boxes = decode(loc.squeeze(0), anchors, variance)
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.squeeze(0), anchors, variance)
        # ignore low scores
        inds = np.where(scores > conf_thre)[0]
        boxes = boxes[inds] * scale
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS, just set the top-k as top-1
        order = scores.argsort()[::-1][:1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_thre)
        dets = dets[keep, :]
        landms = landms[keep].reshape(-1, 2) * np.array([input_size[0], input_size[1]])
        landms = landms.reshape(-1, 10)
        return dets, landms

    input_data = _preprocess(src_image)
    start_time = time.time()
    loc, conf, landms = ort_session.run(None, {input_name: input_data})
    print("inference time:{}".format(time.time() - start_time))
    dets, landms = _post_process(loc, conf, landms)
    filtered_dets, filtered_landms = [], []
    for det, lms in zip(dets, landms):
        if det[4] < 0.8:
            continue
        filtered_dets.append(det)
        filtered_landms.append(lms)
    return np.array(filtered_dets), np.array(filtered_landms)


if __name__ == '__main__':
    onnx_path = "weights/mbnv3_320x180.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (img_size[1], img_size[0]))
        if not ret:
            break
    
        dets, landms = detect_face_lms(frame, ort_session, input_size=(img_size[1], img_size[0]))
        if len(dets) < 1: # no face detected
            continue
        # pdb.set_trace()
        rot_vec, trt_vec = pose_estimate(landms[0].reshape(-1, 2), img_size=img_size)

        print('-------------------------\n', rot_vec, '||||\n', trt_vec,
              '\n-------------------------\n')

        # visualize the results
        for det, lms in zip(dets, landms):
            text = "{:.4f}".format(det[4])
            box = list(map(int, det[:4]))
            cx, cy = box[0], box[1] + 12
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            lms = lms.astype(np.int32).reshape(-1, 2)
            for l in lms.tolist():
                cv2.circle(frame, (l[0], l[1]), 1, (0, 0, 255), 4)
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(-1)
        if key == 27:
            exit(-1)