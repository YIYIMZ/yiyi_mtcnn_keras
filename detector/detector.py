# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:50:15 2018

@author: yy
"""
import os
import sys
sys.path.append(".")
import numpy as np
import cv2
from mtcnn_model.mtcnn_model import p_net, o_net, r_net
from utils import load_weights, process_image, generate_bbox, py_nms, bbox_2_square
from config import LABEL_MAP, MODEL_WEIGHT_SAVE_DIR, NET_SIZE

class Detector:
    def __init__(self, weight_dir,
                 slide_window=False,
                 stride=2,
                 min_face_size=24,
                 threshold=None,   #概率大于threshold的bbox才用
                 scale_factor=0.65,
                 mode=3):
        #mode 
        #1:用p_net 生成r_net的数据
        #2:用p_net r_net 生成 o_net的数据
        #3:用p_net r_net o_net 最后生成结果
        assert mode in [1, 2, 3]
        #实现图片金字塔
        assert scale_factor < 1
        
        #暂时没有使用
        self.slide_window = slide_window
        self.stride = stride
        
        self.mode = mode
        #图片金字塔,图片不能小于这个
        self.min_face_size = min_face_size
        #概率大于threshold的bbox才用
        self.threshold = [0.6, 0.7, 0.7] if threshold is None else threshold
        #实现图片金字塔,以这个比例缩小图片
        self.scale_factor = scale_factor

        self.p_net = None
        self.r_net = None
        self.o_net = None
        self.init_network(weight_dir)

    def init_network(self, weight_dir = MODEL_WEIGHT_SAVE_DIR):
        p_weights, r_weights, o_weights = load_weights(weight_dir)
        print('PNet weight file is: {}'.format(p_weights))
        self.p_net = p_net()
        self.p_net.load_weights(p_weights)
        if self.mode > 1:
            self.r_net = r_net()
            self.r_net.load_weights(r_weights)
        if self.mode > 2:
            self.o_net = o_net()
            self.o_net.load_weights(o_weights)

    def predict(self, image):
        im_ = np.array(image)
        if self.mode == 1:
            return self.predict_with_p_net(im_)
        elif self.mode == 2:
            return self.predict_with_pr_net(im_)
        elif self.mode == 3:
            return self.predict_with_pro_net(im_)
        else:
            raise NotImplementedError('Not implemented yet')
            
    def predict_with_p_net(self, im):
        return self.detect_with_p_net(self, im)
    

    def predict_with_pr_net(self, im):
        boxes, boxes_c, landmark = self.detect_with_p_net(im)
        return self.detect_with_r_net(im, boxes_c)

    def predict_with_pro_net(self, im):
        boxes, boxes_c, landmark = self.detect_with_p_net(im)
        boxes, boxes_c, landmark = self.detect_with_r_net(im, boxes_c)
        return self.detect_with_o_net(im,boxes_c)

    
    def detect_with_p_net(self, im):
        print('p_net_predict---')
        net_size = 12
        current_scale = float(net_size) / self.min_face_size  # find initial scale
        
        im_resized = process_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = []
        while min(current_height, current_width) > net_size:
            
            inputs = np.array([im_resized])
            print('inputs shape: {}'.format(inputs.shape))
            labels, bboxes, landmarks = self.p_net.predict(inputs)
            labels = np.squeeze(labels)
            bboxes = np.squeeze(bboxes)
			#概率大于threshold的bbox才用
            print('labels',labels.shape)
            print('bboxes',bboxes.shape)
            boxes = generate_bbox(labels[:, :, 1], bboxes, current_scale, self.threshold[0])
			#实现图片金字塔
            current_scale *= self.scale_factor
            im_resized = process_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            keep = py_nms(boxes[:, :5], 0.7, 'union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        return self.refine_bboxes(all_boxes)

    
        
    def detect_with_r_net(self, im, dets):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        cls_scores, reg, _ = self.r_net.predict(cropped_ims)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.threshold[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            #landmark = landmark[keep_inds]
        else:
            return None, None, None
        
        
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c,None

    def detect_with_o_net(self, im, dets):
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            
        cls_scores, reg,landmark = self.o_net.predict(cropped_ims)
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.threshold[2])[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None
        
        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T        
        boxes_c = self.calibrate_box(boxes, reg)
        
        boxes = boxes[py_nms(boxes, 0.6, "minimum")]
        keep = py_nms(boxes_c, 0.6, "minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c,landmark
    
    @staticmethod
    def refine_bboxes(all_boxes):
        all_boxes = np.vstack(all_boxes)
        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.5, 'union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return boxes, boxes_c, None

    def calibrate_box(self, bbox, reg):

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c
    
    def convert_to_square(self, bbox):
       
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox
    
    def pad(self, bboxes, w, h):

        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    