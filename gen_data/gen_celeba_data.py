# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:43:19 2018

@author: yy
"""
import os
import random
import sys
sys.path.append(".")

import cv2
import numpy as np
import numpy.random as npr
import config
from config import NET_SIZE, NET_NAMES, GAN_DATA_ROOT_DIR, CELEBA_IMG_DIR, CELEBA_ANNO_LANDMARKS_FILE, CELEBA_ANNO_BBOX_FILE
from utils import resize, iou, flip, rotate, convert_bbox


def generate_data(input_net_name, augmentation=False):
    net_name = input_net_name
    assert net_name in NET_NAMES
    size = NET_SIZE[net_name]
    read_cnn_face_points(net_name)
    #data = read_cnn_face_points(net_name)
    #face_images, face_landmarks = process_data(data, size, augmentation)
    #save_data(face_images, face_landmarks, net_name)
    
def read_cnn_face_points(net_name):
    
    if net_name == 'p_net':
        CELEBA_NUM   = config.PNET_CELEBA_NUM
    if net_name == 'r_net':
        CELEBA_NUM   = config.RNET_CELEBA_NUM
    if net_name == 'o_net':
        CELEBA_NUM   = config.ONET_CELEBA_NUM
        
    landmark_annotation_file = CELEBA_ANNO_LANDMARKS_FILE
    bbox_annotation_file = CELEBA_ANNO_BBOX_FILE
    data = []
    with open(landmark_annotation_file, 'r', encoding='utf-8') as landmarkf:
        landmark_lines = landmarkf.readlines()
        with open(bbox_annotation_file, 'r', encoding='utf-8') as bboxf:
            bboxf_lines = bboxf.readlines()
            
            index = 0
            for bbox_line, landmark_line in zip(bboxf_lines, landmark_lines):
                index += 1
                if index <= 2:
                    continue
                
                if index > CELEBA_NUM:
                    break
                bbox_line = bbox_line.strip()
                bbox_splits = bbox_line.split(' ')
                image_file = bbox_splits[0]
                
                bbox_arr = [ item for item in bbox_splits[1:] if item != '']
                bbox_np = np.array(bbox_arr, dtype=np.int)
                
                x1, y1, w, h = bbox_np[0], bbox_np[1], bbox_np[2], bbox_np[3]
                bbox = convert_bbox((x1, y1, w, h), False)
            
                landmark_line = landmark_line.strip()
                landmark_splits = landmark_line.split(' ')
                
                landmark_arr = [ item for item in landmark_splits[1:] if item != '']
                landmark_np = np.array(landmark_arr, dtype=np.int)
                
                landmark = [(float(landmark_np[2 * i]), float(landmark_np[2 * i + 1])) for i in range(0, 5)]
    
                #print('path---:',os.path.join(CELEBA_IMG_DIR, image_file))
                #print('bbox---:',bbox)
                #print('landmark---:',landmark)
                data.append({
                    'image_file': os.path.join(CELEBA_IMG_DIR, image_file),
                    'bbox': bbox,
                    'landmark': landmark
                })
                
                if len(data) % 1000 == 0:
                    print(" 1000 images done", index)
                    size = NET_SIZE[net_name]
                    face_images, face_landmarks = process_data(data, size, True)
                    save_data(face_images, face_landmarks, net_name, index)
                    data = []
                   
    #return data

def process_data(data, size, augmentation):
    face_images = []
    face_landmarks = []
    idx = 0
    for d in data:
        image_file = d['image_file']
        bbox = d['bbox']
        points = d['landmark']

        img = cv2.imread(image_file)
        im_h, im_w, _ = img.shape
        x1, y1, x2, y2 = bbox
        face_roi = img[y1:y2, x1:x2]
        face_roi = resize(face_roi, size)

        # 归一化
        landmark = normalize_landmark(points, x1, x2, y1, y2)
        face_images.append(face_roi)
        face_landmarks.append(np.array(landmark).reshape(10))

        #数据增强
        if augmentation:
            idx = idx + 1
            
            # gt's width
            bbox_w = x2 - x1 + 1
            # gt's height
            bbox_h = y2 - y1 + 1
            if max(bbox_w, bbox_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size, nx1, nx2, ny1, ny2 = new_bbox(bbox_h, bbox_w, x1, y1)
                if nx2 > im_w or ny2 > im_h:
                    continue

                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]

                _iou = iou(crop_box, np.expand_dims(bbox, 0))

                if _iou > 0.65:
                    resize_im = resize(cropped_im, size)
                    face_images.append(resize_im)
                    # normalize
                    landmark = normalize_landmark2(bbox_size, points, nx1, ny1)
                    face_landmarks.append(landmark.reshape(10))

                    landmark_ = landmark.reshape(-1, 2)

                    nbbox = nx1, ny1, nx2, ny2

                    # mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resize_im, landmark_)
                        face_flipped = resize(face_flipped, size)
                        # c*h*w
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if random.choice([0, 1]) > 0:
                        _landmark = reproject_landmark(nbbox, landmark_)
                        face_rotated_by_alpha, landmark_rotated = rotate(img, nbbox, _landmark, 5)
                        # landmark_offset
                        landmark_rotated = project_landmark(nbbox, landmark_rotated)

                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        face_images.append(face_rotated_by_alpha)
                        face_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))

                    # inverse clockwise rotation
                    if random.choice([0, 1]) > 0:
                        _landmark = reproject_landmark(nbbox, landmark_)
                        face_rotated_by_alpha, landmark_rotated = rotate(img, nbbox, _landmark, -5)  # 顺时针旋转

                        landmark_rotated = project_landmark(nbbox, landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        face_images.append(face_rotated_by_alpha)
                        face_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        face_images.append(face_flipped)
                        face_landmarks.append(landmark_flipped.reshape(10))

    return face_images, face_landmarks


def project(bbox, point):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = (point[0] - x1) / w
    y = (point[1] - y1) / h
    return np.asarray([x, y])


def project_landmark(bbox, landmark):
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = project(bbox, landmark[i])
    return p


def reproject(bbox, point):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w * point[0]
    y = y1 + h * point[1]
    return np.asarray([x, y])


def reproject_landmark(bbox, landmark):
    p = np.zeros((len(landmark), 2))
    for i in range(len(landmark)):
        p[i] = reproject(bbox, landmark[i])
    return p


def normalize_landmark2(bbox_size, points, x1, y1):
    landmark = np.zeros((5, 2))
    for i, point in enumerate(points):
        rv = ((point[0] - x1) / bbox_size, (point[1] - y1) / bbox_size)
        landmark[i] = rv
    return landmark


def normalize_landmark(points, x1, x2, y1, y2):
    landmark = np.zeros((5, 2))
    for i, point in enumerate(points):
        rv = (point[0] - x1) / (x2 - x1), (point[1] - y1) / (y2 - y1)
        landmark[i] = rv
    return landmark


def new_bbox(bbox_h, bbox_w, x1, y1):
    bbox_size = npr.randint(int(min(bbox_w, bbox_h) * 0.8), np.ceil(1.25 * max(bbox_w, bbox_h)))
    delta_x = npr.randint(-bbox_w * 0.2, bbox_w * 0.2)
    delta_y = npr.randint(-bbox_h * 0.2, bbox_h * 0.2)
    nx1 = int(max(x1 + bbox_w / 2 - bbox_size / 2 + delta_x, 0))
    ny1 = int(max(y1 + bbox_h / 2 - bbox_size / 2 + delta_y, 0))
    nx2 = nx1 + bbox_size
    ny2 = ny1 + bbox_size
    return bbox_size, nx1, nx2, ny1, ny2


def save_data(face_images, face_landmarks, net_name, index_start = 0, out_dir = GAN_DATA_ROOT_DIR):
    im_count = index_start
    
    target_size = NET_SIZE[net_name]
    
    save_dir = '{}/{}'.format(out_dir, net_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    img_dir = save_dir + '/landmarks'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    output_file = os.path.join(save_dir, 'landmarks_' + str(target_size) + '.txt')
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for im, point in zip(face_images, face_landmarks):
            if np.sum(np.where(point <= 0, 1, 0)) > 0:
                continue

            if np.sum(np.where(point >= 1, 1, 0)) > 0:
                continue
            
            im_f = os.path.join(img_dir, '%s.jpg' % im_count)
            #print('processing {}'.format(im_f))
            cv2.imwrite(im_f, im)

            txt = '{} -2 {}\n'.format(im_f, ' '.join(map(str, list(point))))
            f.write(txt)

            im_count += 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s The specific net, p_net, r_net, or o_net \r\n" % (sys.argv[0]))
    else:
        generate_data(sys.argv[1],True)

