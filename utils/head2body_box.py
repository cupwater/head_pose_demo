'''
Author: Peng Bo
Date: 2022-09-18 11:23:32
LastEditTime: 2022-09-18 21:59:30
Description: 

'''
import numpy as np
import pdb

def head2body_box(img, head_box):
    # Todo
    # here we just return head_box, 
    center_x = (head_box[0]+head_box[2]) / 2
    center_y = (head_box[1]+head_box[3]) / 2
    width = min(head_box[2]-head_box[0], head_box[3]-head_box[1])
    body_x1 = int(max(center_x - width*3, 0))
    body_x2 = int(min(center_x + width*3, img.shape[1]))
    body_y1 = int(max(center_y - width*1, 0))
    body_y2 = int(min(center_y + width*9, img.shape[0]))
    return img[body_y1:body_y2, body_x1:body_x2]