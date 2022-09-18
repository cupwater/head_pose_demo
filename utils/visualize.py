'''
Author: Peng Bo
Date: 2022-09-18 10:15:06
LastEditTime: 2022-09-19 01:13:39
Description: 

'''
import numpy as np

def get_landmarks_from_heatmap(pred_heatmap, src_size, tgt_size=(384,288)):
    pred_heatmap = np.transpose(pred_heatmap, [1,2,0])
    h,w,_ = pred_heatmap.shape
    ## get the position of landmarks
    x_idxs = np.array(range(w)).reshape(1, -1)
    x_idxs = np.repeat(x_idxs, h, axis=0)
    y_idxs = np.array(range(h)).reshape(-1, 1)
    y_idxs = np.repeat(y_idxs, w, axis=1)
    
    landmarks = []
    for i in range(pred_heatmap.shape[-1]):
        x_pos = int(np.sum(pred_heatmap[:,:,i] * x_idxs) / np.sum(pred_heatmap[:,:,i]))
        y_pos = int(np.sum(pred_heatmap[:,:,i] * y_idxs) / np.sum(pred_heatmap[:,:,i]))
        landmarks.append([x_pos, y_pos])
    landmarks = np.array(landmarks)
    
    if 1.0*src_size[0]/tgt_size[0] > 1.0*src_size[1]/tgt_size[1]:
        landmarks[:,0] = landmarks[:,0]/w * tgt_size[1] * src_size[0] / tgt_size[0]
        landmarks[:,1] = landmarks[:,1]/h * src_size[0]
    else:
        landmarks[:,0] = landmarks[:,0]/w * src_size[1]
        landmarks[:,1] = landmarks[:,1]/h * tgt_size[0] * src_size[1] / tgt_size[1]

    landmarks = landmarks.astype(np.int32)
    return landmarks

# def draw_landmarks(image, landmarks):
#     # img = np.transpose(input.cpu().numpy())[:,:,0]
#     img = input[0,0,:,:]
#     print(img.shape)
#     img = 255*(img-np.min(img)) / (np.max(img) - np.min(img))
#     img = img.astype(np.uint8)
#     img = cv2.merge([img, img, img])
#     # draw landmarks on image
#     for (x_pos,y_pos) in landmarks:
#         cv2.circle(img, (x_pos,y_pos), 2, (0,0,255), -1)
#     return img

