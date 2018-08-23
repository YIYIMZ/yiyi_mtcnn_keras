# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:10:21 2018

@author: yy
"""

import os
import sys
sys.path.append(".")
import cv2
import numpy as np
import pickle
from detector.detector import Detector
from config import NET_SIZE, NET_NAMES, WIDER_FACE_IMG_DIR, WIDER_FACE_ANNO_FILE, GAN_DATA_ROOT_DIR, MODEL_WEIGHT_SAVE_DIR, DETECT_WIDEFACE_NUM
from utils import bbox_2_square, iou, save_dict_to_hdf5, load_dict_from_hdf5

def build_save_path(out_dir):
    neg_dir = os.path.join(out_dir, 'negative')
    pos_dir = os.path.join(out_dir, 'positive')
    part_dir = os.path.join(out_dir, 'part')
    for file_dir in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
    return neg_dir, pos_dir, part_dir


def save_hard_examples(net_name, out_dir, dataset, detections_path):
    #图片位置
    neg_dir, pos_dir, part_dir = build_save_path(out_dir)
    #文件位置
    target_size = NET_SIZE[net_name]
    pos_file = open(os.path.join(out_dir, 'pos_' + str(target_size) + '.txt'), 'w')
    neg_file = open(os.path.join(out_dir, 'neg_' + str(target_size) + '.txt'), 'w')
    part_file = open(os.path.join(out_dir, 'part_' + str(target_size) + '.txt'), 'w')
    
    with open(detections_path, 'rb') as f:
        detections = pickle.load(f)
    
    #images_path images_jpg bboxes
    #image_files = dataset['images']
    image_files = dataset['images_jpg']
    bboxes_true = dataset['bboxes']
    
    bboxes_pred = detections['bboxes']
    print('save_hard_examples begin---')
    im_size = NET_SIZE[net_name]
    n_idx = 0
    p_idx = 0
    d_idx = 0

    for im_f, bbox_pred, box_true in zip(image_files, bboxes_pred, bboxes_true):
        if bbox_pred.shape[0] == 0:
            continue
        box_true = np.array(box_true, dtype=np.float32).reshape(-1, 4)
        #print('box_true---:',box_true)
        #print('bbox_pred---:',bbox_pred.shape)
        img = im_f #cv2.imread(im_f)
        bbox_pred = bbox_2_square(bbox_pred)
        bbox_pred[:, 0:4] = np.round(bbox_pred[:, 0:4])
        #print('bbox_pred---:',bbox_pred.shape)
        neg_num = 0
        for box in bbox_pred:
            #print('box---:',box)
            #print('box---:',box.shape)
            x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            if width < 20 or x1 < 0 or y1 < 0 or x2 > img.shape[1] - 1 or y2 > img.shape[0] - 1:
                continue
            _iou = iou(box, box_true)
            cropped_im = img[y1:y2 + 1, x1:x2 + 1, :]
            resized_im = cv2.resize(cropped_im, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
            if np.max(_iou) < 0.3 and neg_num < 60:
                file_name = os.path.join(neg_dir, '{0:08}.jpg'.format(n_idx))
                cv2.imwrite(file_name, resized_im)
                neg_file.write('{} 0\n'.format(file_name))
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(_iou)
                assigned_gt = box_true[idx]
                x1t, y1t, x2t, y2t = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x1t) / float(width)
                offset_y1 = (y1 - y1t) / float(height)
                offset_x2 = (x2 - x2t) / float(width)
                offset_y2 = (y2 - y2t) / float(height)

                if np.max(_iou) >= 0.65:
                    file_name = os.path.join(pos_dir, '{0:08}.jpg'.format(p_idx))
                    pos_file.write(
                        file_name + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(file_name, resized_im)
                    p_idx += 1

                elif np.max(_iou) >= 0.4:
                    file_name = os.path.join(part_dir, '{0:08}.jpg'.format(d_idx))
                    part_file.write(file_name + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(file_name, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()

def load_widerface_dataset(images_dir = WIDER_FACE_IMG_DIR, anno_file = WIDER_FACE_ANNO_FILE):
    data = dict()
    images_path = []
    images_jpg = []
    bboxes = []
    
    num = 0
    with open(anno_file, 'r', encoding='utf-8')as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            image_fall_path = images_dir+'/'+ line
            images_path.append(image_fall_path)
            
            im = cv2.imread(image_fall_path)
            images_jpg.append(im)
            face_num = int(f.readline().strip())
            faces = []
            for i in range(face_num):
                bb_info = f.readline().strip('\n').split(' ')
                x1, y1, w, h = [float(bb_info[i]) for i in range(4)]
                faces.append([x1, y1, x1 + w, y1 + h])
            bboxes.append(faces)
            
            num += 1
            if num >= DETECT_WIDEFACE_NUM:
                break
        
        data['images_path'] = images_path  # all image pathes
        data['images_jpg'] = images_jpg  # all image pathes
        data['bboxes'] = bboxes  # all image bboxes

    return data

def main(input_net_name):
    net_name = input_net_name
    assert net_name in NET_NAMES
    images_dir = WIDER_FACE_IMG_DIR
    annotation_file = WIDER_FACE_ANNO_FILE
    out_dir = GAN_DATA_ROOT_DIR
    out_dir = '{}/{}'.format(out_dir, net_name)
    
    if net_name == 'r_net':
        mode = 1
    elif net_name == 'o_net':
        mode = 2
    
    #images_path images_jpg bboxes
    dataset = load_widerface_dataset(images_dir, annotation_file)

    detector = Detector(weight_dir= MODEL_WEIGHT_SAVE_DIR, mode=mode)
    
    bboxes_all = []
    
    #p_net 一次测一张图片，注意，其返回可能会有多个，因为图片中可以包含多张面，而且还有图片金字塔
    print('data img len --:',len(dataset['images_jpg']))
    for img_jpg in dataset['images_jpg']:
        _, bboxes, _ = detector.predict(img_jpg)
        bboxes_all.append(bboxes)
    
    bboxes_all = np.array(bboxes_all)
    print('predict over---')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    detections_path = os.path.join(out_dir, 'middle_wideface_data_det.pkl')
    with open(detections_path, 'wb') as f:
        pickle.dump({
            'bboxes': bboxes_all,
        }, f)
    
    save_hard_examples(net_name, out_dir, dataset, detections_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s The specific net, r_net, or o_net \r\n" % (sys.argv[0]))
    else:
        main(sys.argv[1])
