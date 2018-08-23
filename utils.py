# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:31:51 2018

@author: yy
"""
import h5py
import cv2
import numpy as np
import glob

def resize(im, target_size):
    h, w, ch = im.shape
    if h != target_size or w != target_size:
        im = cv2.resize(im, (target_size, target_size))
    return im


def flip(face, landmark):
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return face_flipped_by_x, landmark_


def rotate(img, bbox, landmark, alpha):

    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)

    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)

    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[y1:y2 + 1, x1:x2 + 1]
    return face, landmark_


def bbox_2_square(bbox):

    print('bbox_2_square---:',bbox.shape)
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox



def convert_bbox(box, kind=True):
    """
        (x1, y1, x2, y2) --> (x1, y1, w, h) (kind=True)
        or
        (x1, y1, w, h) --> (x1, y1, x2, y2) (kind=False)
    """
    a, b, c, d = box
    if kind:
        return (a, b, c - a + 1, d - b + 1)
    else:
        return (a, b, a + c - 1, b + d - 1)

#h5文件不能存放字典、NONE，可以用pkl
def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, list, tuple)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def load_weights(weights_dir):
    weights_files = glob.glob('{}/*.h5'.format(weights_dir))
    p_net_weight = None
    r_net_weight = None
    o_net_weight = None
    for wf in weights_files:
        if 'p_net' in wf:
            p_net_weight = wf
        elif 'r_net' in wf:
            r_net_weight = wf
        elif 'o_net' in wf:
            o_net_weight = wf
        else:
            raise ValueError('No valid weights files !')

    if p_net_weight is None and r_net_weight is None and o_net_weight is None:
        raise ValueError('No valid weights files !')

    return p_net_weight, r_net_weight, o_net_weight


def process_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return img_resized


def batch_gen_bbox(cls_map, reg, scale, threshold, stride=2, cell_size=12):
    bboxes = []
    for cls, bbox in zip(cls_map, reg):
        b = generate_bbox(cls, bbox, scale, threshold, stride, cell_size)
        bboxes.append(b)
    return bboxes


def generate_bbox(cls_map, reg, scale, threshold, stride=2, cell_size=12):

    t_index = np.where(cls_map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    # offset
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    bbox = np.vstack([np.round((stride * t_index[1]) / scale),
                      np.round((stride * t_index[0]) / scale),
                      np.round((stride * t_index[1] + cell_size) / scale),
                      np.round((stride * t_index[0] + cell_size) / scale),
                      score,
                      reg])

    return bbox.T

def py_nms(bboxes, thresh, mode="union"):
    assert mode in ['union', 'minimum']

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        else:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        # keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def iou(box, boxes):
#    print('iou----------:',boxes.shape)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


