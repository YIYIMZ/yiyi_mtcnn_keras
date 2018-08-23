# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 10:28:23 2018

@author: yy
"""
import os
import sys
sys.path.append(".")
import cv2
import numpy as np
import numpy.random as npr
import utils
import config
from config import NET_SIZE, NET_NAMES, WIDER_FACE_IMG_DIR, WIDER_FACE_ANNO_FILE, GAN_DATA_ROOT_DIR
#from utils import iou, convert_bbox

CURR_DIR = os.path.dirname(__file__)


def main(input_net_name):
    net_name = input_net_name
    assert net_name in config.NET_NAMES
    
    if net_name == 'p_net':
        BACKGRAND_NEG_NUM   = config.PNET_BACKGRAND_NEG_NUM
        FACE_NEG_NUM        = config.PNET_FACE_NEG_NUM
        POS_PART_NUM        = config.PNET_POS_PART_NUM
        WIDEFACE_NUM        = config.PNET_WIDEFACE_NUM
    if net_name == 'r_net':
        BACKGRAND_NEG_NUM   = config.RNET_BACKGRAND_NEG_NUM
        FACE_NEG_NUM        = config.RNET_FACE_NEG_NUM
        POS_PART_NUM        = config.RNET_POS_PART_NUM  
        WIDEFACE_NUM        = config.RNET_WIDEFACE_NUM
    if net_name == 'o_net':
        BACKGRAND_NEG_NUM   = config.ONET_BACKGRAND_NEG_NUM
        FACE_NEG_NUM        = config.ONET_FACE_NEG_NUM
        POS_PART_NUM        = config.ONET_POS_PART_NUM
        WIDEFACE_NUM        = config.ONET_WIDEFACE_NUM

    images_dir = config.WIDER_FACE_IMG_DIR
    anno_file = config.WIDER_FACE_ANNO_FILE
    out_dir = config.GAN_DATA_ROOT_DIR

    target_size = config.NET_SIZE[net_name]
    save_dir = '{}/{}'.format(out_dir, net_name)

    pos_save_dir = save_dir + '/positive'
    part_save_dir = save_dir + '/part'
    neg_save_dir = save_dir + '/negative'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    f1 = open(os.path.join(save_dir, 'pos_' + str(target_size) + '.txt'), 'a')
    f2 = open(os.path.join(save_dir, 'neg_' + str(target_size) + '.txt'), 'a')
    f3 = open(os.path.join(save_dir, 'part_' + str(target_size) + '.txt'), 'a')
    
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
		
    n_lines = len(annotations)
    print('%d pics in total' % n_lines)
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # dont care
    idx = 0
    box_idx = 0
    
    idx_line = 0
    while idx_line < n_lines:
        #获取图片及bbox
        image_name = annotations[idx_line].strip()
        assert '/' in image_name
        
        n_faces = int(annotations[idx_line + 1])
        
        
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        
        bboxes = []
        for i in range(n_faces):
            anno = annotations[idx_line + 2 + i].strip().split()
            anno = list(map(int, anno))
            x1, y1, w, h = anno[0], anno[1], anno[2], anno[3]
            box = utils.convert_bbox((x1, y1, w, h), False)
            bboxes.append(box)
        bboxes = np.array(bboxes, dtype=np.float32)
        
        idx_line += 1 + n_faces + 1
        
        #生成样本
        idx += 1
        if idx % 1000 == 0:
            print(idx, 'images done')
        if idx > WIDEFACE_NUM:
            break
        height, width, channel = img.shape

        #随机生成 negative 样本，
        neg_num = 0
        while neg_num < BACKGRAND_NEG_NUM:
            size = npr.randint(target_size, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])

            _iou = utils.iou(crop_box, bboxes)

            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            if np.max(_iou) < 0.3:
                # _iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write(save_dir + '/negative/%s' % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            print('{} images done, pos: {},  part: {},  neg: {}'.format(idx, p_idx, d_idx, n_idx))

        #以标记的box为中心，分别生成 negative 5个、positive 和 part 共20个 三种样本，
        for box in bboxes:
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue

            # 生成 negative
            for i in range(FACE_NEG_NUM):
                size = npr.randint(target_size, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                delta_x = npr.randint(max(-size, -x1), w)
                delta_y = npr.randint(max(-size, -y1), h)
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                _iou = utils.iou(crop_box, bboxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                
                #neg iou 小于0.3
                if np.max(_iou) < 0.3:
                    # _iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_dir + "/negative/%s" % n_idx + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # pos 和 part 样本
            for i in range(POS_PART_NUM):
                size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                delta_x = npr.randint(-w * 0.2, w * 0.2)
                delta_y = npr.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv2.resize(cropped_im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                if utils.iou(crop_box, box_) >= 0.65:
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write(save_dir + '/positive/%s' % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))

                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif utils.iou(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write(save_dir + '/part/%s' % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print('{} images done, pos: {},  part: {},  neg: {}'.format(idx, p_idx, d_idx, n_idx))

    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s The specific net, p_net, r_net, or o_net \r\n" % (sys.argv[0]))
    else:
        main(sys.argv[1])
    
    


