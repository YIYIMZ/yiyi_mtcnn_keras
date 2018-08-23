# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:02:01 2018

@author: yy
"""
import os
import sys
sys.path.append(".")

from config import NET_SIZE, BATCH_SIZE, ONET_EPOCHS, ONET_LEARNING_RATE, GAN_DATA_ROOT_DIR

from load_data import load_dataset, DataGenerator
from train_pub import create_callbacks_model_file, train_o_net_with_data_generator

def train_with_data_generator(dataset_root_dir = GAN_DATA_ROOT_DIR, weights_file=None):
    net_name = 'o_net'
    batch_size = BATCH_SIZE
    epochs = ONET_EPOCHS
    learning_rate = ONET_LEARNING_RATE
    
    dataset_dir = os.path.join(dataset_root_dir, net_name)
    pos_dataset_path = os.path.join(dataset_dir, 'pos_shuffle.h5')
    neg_dataset_path = os.path.join(dataset_dir, 'neg_shuffle.h5')
    part_dataset_path = os.path.join(dataset_dir, 'part_shuffle.h5')
    landmarks_dataset_path = os.path.join(dataset_dir, 'landmarks_shuffle.h5')

    data_generator = DataGenerator(pos_dataset_path, neg_dataset_path, part_dataset_path, landmarks_dataset_path, batch_size,
                                   im_size=NET_SIZE['o_net'])
    data_gen = data_generator.generate()
    steps_per_epoch = data_generator.steps_per_epoch()

    callbacks, model_file = create_callbacks_model_file(net_name, epochs)

    _o_net = train_o_net_with_data_generator(data_gen, steps_per_epoch,
                                             initial_epoch=0,
                                             epochs=epochs,
                                             lr=learning_rate,
                                             callbacks=callbacks,
                                             weights_file=weights_file)
    _o_net.save_weights(model_file)

if __name__ == '__main__':

    train_with_data_generator()