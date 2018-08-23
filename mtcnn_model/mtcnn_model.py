# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:44:08 2018

@author: yy
"""

from keras import Input, layers, Model

def p_net(training=False):
    x = Input(shape=(12, 12, 3)) if training else Input(shape=(None, None, 3))
    y = layers.Conv2D(10, 3, padding='valid', strides=(1, 1), name='p_conv1')(x)
    y = layers.PReLU(shared_axes=(1, 2), name='p_prelu1')(y)
    y = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='p_max_pooling1')(y)
    y = layers.Conv2D(16, 3, padding='valid', strides=(1, 1), name='p_conv2')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='p_prelu2')(y)
    y = layers.Conv2D(32, 3, padding='valid', strides=(1, 1), name='p_conv3')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='p_prelu3')(y)

    classifier = layers.Conv2D(2, 1, activation='softmax', name='p_classifier')(y)
    bbox = layers.Conv2D(4, 1, name='p_bbox')(y)
    landmark = layers.Conv2D(10, 1, padding='valid', name='p_landmark')(y)

    if training:
        classifier = layers.Reshape((2,), name='p_classifier1')(classifier)
        bbox = layers.Reshape((4,), name='p_bbox1')(bbox)
        landmark = layers.Reshape((10,), name='p_landmark1')(landmark)
        outputs = layers.concatenate([classifier, bbox, landmark])
        model = Model(inputs=[x], outputs=[outputs], name='P_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox, landmark], name='P_Net')
    return model


def r_net(training=False):
    x = Input(shape=(24, 24, 3))
    y = layers.Conv2D(28, 3, padding='same', strides=(1, 1), name='r_conv1')(x)
    y = layers.PReLU(shared_axes=(1, 2), name='r_prelu1')(y)
    y = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='p_max_pooling1')(y)
    y = layers.Conv2D(48, 3, padding='valid', strides=(1, 1), name='r_conv2')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='r_prelu2')(y)
    y = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='p_max_pooling2')(y)
    y = layers.Conv2D(64, 2, padding='valid', name='r_conv3')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='r_prelu3')(y)
    y = layers.Dense(128, name='r_dense')(y)
    y = layers.PReLU(name='r_prelu4')(y)
    y = layers.Flatten()(y)

    classifier = layers.Dense(2, activation='softmax', name='r_classifier')(y)
    bbox = layers.Dense(4, name='r_bbox')(y)
    landmark = layers.Dense(10, name='r_landmark')(y)

    if training:
        outputs = layers.concatenate([classifier, bbox, landmark])
        model = Model(inputs=[x], outputs=[outputs], name='R_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox, landmark], name='R_Net')

    return model


def o_net(training=False):
    x = Input(shape=(48, 48, 3))
    y = layers.Conv2D(32, 3, padding='same', strides=(1, 1), name='o_conv1')(x)
    y = layers.PReLU(shared_axes=(1, 2), name='o_prelu1')(y)
    y = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='o_max_pooling1')(y)
    y = layers.Conv2D(64, 3, padding='valid', strides=(1, 1), name='o_conv2')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='o_prelu2')(y)
    y = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='o_max_pooling2')(y)
    y = layers.Conv2D(64, 3, padding='valid', strides=(1, 1), name='o_conv3')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='o_prelu3')(y)
    y = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='o_max_pooling3')(y)
    y = layers.Conv2D(128, 2, padding='valid', strides=(1, 1), name='o_conv4')(y)
    y = layers.PReLU(shared_axes=(1, 2), name='o_prelu4')(y)
    y = layers.Dense(256, name='o_dense')(y)
    y = layers.PReLU(name='o_prelu5')(y)
    y = layers.Flatten()(y)

    classifier = layers.Dense(2, activation='softmax', name='o_classifier')(y)
    bbox = layers.Dense(4, name='o_bbox')(y)
    landmark = layers.Dense(10, name='o_landmark')(y)

    if training:
        outputs = layers.concatenate([classifier, bbox, landmark])
        model = Model(inputs=[x], outputs=[outputs], name='O_Net')
    else:
        model = Model(inputs=[x], outputs=[classifier, bbox, landmark], name='O_Net')
    return model


if __name__ == '__main__':
    p = p_net()
    p.summary()

    r = r_net()
    r.summary()

    o = o_net()
    o.summary()
