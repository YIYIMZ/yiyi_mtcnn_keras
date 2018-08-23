# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:16:56 2018

@author: yy
"""

import os
import random
import sys
sys.path.append(".")
import cv2
import numpy as np

from config import NET_SIZE, NET_NAMES, GAN_DATA_ROOT_DIR
from utils import resize, save_dict_to_hdf5


def main(input_net_name):
    net_name = input_net_name
    assert net_name in NET_NAMES

    target_size = NET_SIZE[net_name]
    net_data_dir = os.path.join(GAN_DATA_ROOT_DIR, net_name)

    with open('%s/pos_%s.txt' % (net_data_dir, target_size), 'r') as f:
        pos = f.readlines()
    with open('%s/neg_%s.txt' % (net_data_dir, target_size), 'r') as f:
        neg = f.readlines()
    with open('%s/part_%s.txt' % (net_data_dir, target_size), 'r') as f:
        part = f.readlines()
    with open('%s/landmarks_%s.txt' % (net_data_dir, target_size), 'r') as f:
        landmark_anno = f.readlines()

    create_pos_dataset(net_name, pos, target_size, net_data_dir)
    create_neg_dataset(net_name, neg, target_size, net_data_dir)
    create_part_dataset(net_name, part, target_size, net_data_dir)
    create_landmark_dataset(net_name, landmark_anno, target_size, net_data_dir)

def create_pos_dataset(net_name, pos, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in pos:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = words[0]
        else:    
            image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('pos data doing, total: {}'.format(len(ims)))
    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    landmark_data_filename = os.path.join(out_dir, 'pos_shuffle.h5')
    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, landmark_data_filename)

    print('pos data done, total: {}'.format(len(ims)))    
    
    
def create_neg_dataset(net_name, neg, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in neg:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = words[0]
        else:    
            image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))
        
        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('neg data doing, total: {}'.format(len(ims)))
    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    landmark_data_filename = os.path.join(out_dir, 'neg_shuffle.h5')
    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, landmark_data_filename)

    print('neg data done, total: {}'.format(len(ims)))    
    
    
def create_part_dataset(net_name, part, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in part:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = words[0]
        else:    
            image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:6]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('part data doing, total: {}'.format(len(ims)))
    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    landmark_data_filename = os.path.join(out_dir, 'part_shuffle.h5')
    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'bbox': landmarks}, landmark_data_filename)

    print('part data done, total: {}'.format(len(ims)))


def create_landmark_dataset(net_name, landmark_anno, target_size, out_dir):
    ims = []
    landmarks = []
    labels = []
    for line in landmark_anno:

        words = line.strip().split()
        if '.jpg' in words[0]:
            image_file_name = words[0]
        else:    
            image_file_name = words[0] + '.jpg'

        im = cv2.imread(image_file_name)
        im = resize(im, target_size)
        im = im.astype('uint8')
        ims.append(im)

        labels.append(int(words[1]))

        landmark = words[2:12]
        landmark = list(map(float, landmark))
        landmark = np.array(landmark, dtype='float32')
        landmarks.append(landmark)
        if len(ims)%500 == 0 :
            print('landmarks data doing, total: {}'.format(len(ims)))
    landmark_data = list(zip(labels, ims, landmarks))
    random.shuffle(landmark_data)
    labels, ims, landmarks = zip(*landmark_data)

    landmark_data_filename = os.path.join(out_dir, 'landmarks_shuffle.h5')
    save_dict_to_hdf5({'labels': labels, 'ims': ims, 'landmarks': landmarks}, landmark_data_filename)

    print('landmarks data done, total: {}'.format(len(ims)))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s The specific net, p_net, r_net, or o_net \r\n" % (sys.argv[0]))
    else:
        main(sys.argv[1])
    
