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


def head2body_box1(img, head_box):
    # Todo
    # here we just return head_box, 
    center_x = (head_box[0]+head_box[2]) / 2
    center_y = (head_box[1]+head_box[3]) / 2
    width = min(head_box[2]-head_box[0], head_box[3]-head_box[1])
    body_x1 = int(max(center_x - width*1.3, 0))
    body_y1 = int(max(center_y - width*1, 0))
    body_x2 = int(min(center_x + width*1.3, img.shape[1]))
    body_y2 = int(min(center_y + width*5, img.shape[0]))
    return [body_x1, body_y1, body_x2, body_y2]


def crop_img_by_box(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0))
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size_w = int(max([w, h])*0.8)
    size_h = int(max([w, h]) * 0.8)
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = cx - size_w//2
    x2 = x1 + size_w
    y1 = cy - int(size_h * 0.4)
    y2 = y1 + size_h

    left = 0
    top = 0
    bottom = 0
    right = 0
    if x1 < 0:
        left = -x1
    if y1 < 0:
        top = -y1
    if x2 >= width:
        right = x2 - width
    if y2 >= height:
        bottom = y2 - height

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))

    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    cropped = image[y1:y2, x1:x2]