'''
Author: Peng Bo
Date: 2022-09-18 11:23:32
LastEditTime: 2022-11-14 18:28:44
Description: 

'''


def head2box(img, head_box):
    # Todo
    _x1 = int(max(head_box[0], 0))
    # _x2 = int(min(head_box[0]+head_box[2], img.shape[1]))
    _y1 = int(max(head_box[1], 0))
    # _y2 = int(min(head_box[1] + head_box[3], img.shape[0]))
    return [_x1, _y1, head_box[2], head_box[3]]

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


