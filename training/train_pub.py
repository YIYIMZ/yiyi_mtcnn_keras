# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:48:47 2018

@author: yy
"""

import datetime
import os
import sys
sys.path.append(".")
import keras.backend.tensorflow_backend as TK
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from mtcnn_model.mtcnn_model import p_net, r_net, o_net
from config import LABEL_MAP, MODEL_WEIGHT_SAVE_DIR, LOG_DIR

MODES = ['label', 'bbox', 'landmark']

NEGATIVE = TK.constant(LABEL_MAP['0'])
POSITIVE = TK.constant(LABEL_MAP['1'])
PARTIAL = TK.constant(LABEL_MAP['-1'])
LANDMARK = TK.constant(LABEL_MAP['-2'])
num_keep_radio = 0.7

def cal_mask(label_true, _type='label'):
    def true_func():
        return 0

    def false_func():
        return 1

    label_true_int32 = tf.cast(label_true, dtype=tf.int32)
    if _type == 'label':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], x[1]), true_func, false_func), label_true_int32)
    elif _type == 'bbox':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.equal(x[0], 1), true_func, false_func), label_true_int32)
    elif _type == 'landmark':
        label_filtered = tf.map_fn(lambda x: tf.cond(tf.logical_and(tf.equal(x[0], 1), tf.equal(x[1], 1)),
                                                     false_func, true_func), label_true_int32)
    else:
        raise ValueError('Unknown type of: {} while calculate mask'.format(_type))

    mask = tf.cast(label_filtered, dtype=tf.int32)
    return mask


def label_ohem(label_true, label_pred):
    label_int = cal_mask(label_true, 'label')

    num_cls_prob = tf.size(label_pred)
    print('num_cls_prob: ', num_cls_prob)
    cls_prob_reshape = tf.reshape(label_pred, [num_cls_prob, -1])
    print('label_pred shape: ', tf.shape(label_pred))
    num_row = tf.shape(label_pred)[0]
    num_row = tf.to_int32(num_row)
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)

    valid_inds = cal_mask(label_true, 'label')
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(tf.cast(num_valid, dtype=tf.float32) * num_keep_radio, dtype=tf.int32)
    # set 0 to invalid sample
    loss = loss * tf.cast(valid_inds, dtype=tf.float32)
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem(label_true, bbox_true, bbox_pred):
    mask = cal_mask(label_true, 'bbox')
    num = tf.reduce_sum(mask)
    keep_num = tf.cast(num, dtype=tf.int32)

    bbox_true1 = tf.boolean_mask(bbox_true, mask, axis=0)
    bbox_pred1 = tf.boolean_mask(bbox_pred, mask, axis=0)

    square_error = tf.square(bbox_pred1 - bbox_true1)
    square_error = tf.reduce_sum(square_error, axis=1)

    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def landmark_ohem(label_true, landmark_true, landmark_pred):
    mask = cal_mask(label_true, 'landmark')
    num = tf.reduce_sum(mask)
    keep_num = tf.cast(num, dtype=tf.int32)

    landmark_true1 = tf.boolean_mask(landmark_true, mask)
    landmark_pred1 = tf.boolean_mask(landmark_pred, mask)

    square_error = tf.square(landmark_pred1 - landmark_true1)
    square_error = tf.reduce_sum(square_error, axis=1)

    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)


def _loss_func(y_true, y_pred):
    labels_true = y_true[:, :2]
    bbox_true = y_true[:, 2:6]
    landmark_true = y_true[:, 6:]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]
    landmark_pred = y_pred[:, 6:]

    label_loss = label_ohem(labels_true, labels_pred)
    bbox_loss = bbox_ohem(labels_true, bbox_true, bbox_pred)
    landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)

    return label_loss + bbox_loss * 0.5 + landmark_loss * 0.5


def _onet_loss_func(y_true, y_pred):
    labels_true = y_true[:, :2]
    bbox_true = y_true[:, 2:6]
    landmark_true = y_true[:, 6:]

    labels_pred = y_pred[:, :2]
    bbox_pred = y_pred[:, 2:6]
    landmark_pred = y_pred[:, 6:]

    label_loss = label_ohem(labels_true, labels_pred)
    bbox_loss = bbox_ohem(labels_true, bbox_true, bbox_pred)
    landmark_loss = landmark_ohem(labels_true, landmark_true, landmark_pred)

    return label_loss + bbox_loss * 0.5 + landmark_loss



def train_p_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _p_net = p_net(training=True)
    _p_net.summary()
    if weights_file is not None:
        _p_net.load_weights(weights_file)

    _p_net.compile(Adam(lr=lr), loss=_loss_func, metrics=['accuracy'])

    _p_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _p_net

def train_r_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _r_net = r_net(training=True)
    _r_net.summary()
    if weights_file is not None:
        _r_net.load_weights(weights_file)

    _r_net.compile(Adam(lr=lr), loss=_loss_func, metrics=['accuracy'])

    _r_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _r_net

def train_o_net_with_data_generator(data_gen, steps_per_epoch, initial_epoch=0, epochs=1000, lr=0.001,
                                    callbacks=None, weights_file=None):
    _o_net = o_net(training=True)
    _o_net.summary()
    if weights_file is not None:
        _o_net.load_weights(weights_file)

    _o_net.compile(Adam(lr=lr), loss=_onet_loss_func, metrics=['accuracy'])

    _o_net.fit_generator(data_gen,
                         steps_per_epoch=steps_per_epoch,
                         initial_epoch=initial_epoch,
                         epochs=epochs,
                         callbacks=callbacks)
    return _o_net


def create_callbacks_model_file(net_name, epochs):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f')
    
    log_dir = "{}/{}/{}".format(LOG_DIR, net_name, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    model_file_path = '{}/{}.h5'.format(MODEL_WEIGHT_SAVE_DIR, net_name+'_weight')

    checkpoint = ModelCheckpoint(model_file_path, verbose=0, save_weights_only=True)
    return [checkpoint, tensor_board], model_file_path

