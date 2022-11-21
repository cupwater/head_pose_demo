'''
Author: Peng Bo
Date: 2022-11-13 22:19:55
LastEditTime: 2022-11-21 11:56:11
Description: 

'''

import time
import cv2
import numpy as np
import onnxruntime as ort

from utils.box_util import decode, decode_landm, gen_bbox
from utils.nms import py_cpu_nms

import pdb

def detect_face_lms(src_image, ort_session, input_size=(320, 180)):    
    input_name = ort_session.get_inputs()[0].name
    def _preprocess(src_image):
        # pre-process the input image 
        input_data = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, input_size)
        input_data = (input_data - np.array([104, 117, 123]))
        target_data = np.expand_dims(np.transpose(input_data, [2, 0, 1]), axis=0)
        target_data = target_data.astype(np.float32)
        return target_data

    input_data = _preprocess(src_image)
    start_time = time.time()
    loc, conf, landms = ort_session.run(None, {input_name: input_data})
    print("inference time:{}".format(time.time() - start_time))
    return loc, conf, landms


if __name__ == '__main__':
    onnx_path = "weights/mbnv3_320x180.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    cap = cv2.VideoCapture(0)
    img_size = (320, 180)
    anchors, variance = gen_bbox(img_size)


    confidence_threshold = 0.6
    nms_threshold = 0.5


    im_size_min = 180
    target_size = 320
    resize = float(target_size) / float(im_size_min)

    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    lms_scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0],
                               img_size[1], img_size[0], img_size[1], img_size[0],
                               img_size[1], img_size[0]])
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, img_size)
        if not ret:
            break
        loc, conf, landms = detect_face_lms(frame, ort_session, input_size=img_size)
        boxes = decode(loc.squeeze(0), anchors, variance)
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landms.squeeze(0), anchors, variance)
        print("debug info")
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds] * scale
        landms = landms[inds] * lms_scale
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        dets = np.concatenate((dets, landms), axis=1)
        pdb.set_trace()
        for box in dets:
            if box[4] < 0.8:
                continue
            text = "{:.4f}".format(box[4])
            box = list(map(int, box))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cx = box[0]
            cy = box[1] + 12
            cv2.putText(frame, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # landms
            cv2.circle(frame, (box[5], box[6]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (box[7], box[8]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (box[9], box[10]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (box[11], box[12]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (box[13], box[14]), 1, (255, 0, 0), 4)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27:
            exit(-1)