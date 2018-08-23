# -*- coding: utf-8 -*-
"""
Created on Web Aug 22 22:30:31 2018

@author: yy
"""

import numpy as np
import sys
sys.path.append(".")
import cv2

from detector.detector import Detector

from config import MODEL_WEIGHT_SAVE_DIR, TEST_INPUT_IMG_DIR, TEST_OUTPUT_IMG_DIR

def main(image_file):
    detector = Detector(weight_dir=MODEL_WEIGHT_SAVE_DIR, mode=3, min_face_size=24)
    input_img_full_path = TEST_INPUT_IMG_DIR + '/' + image_file
    output_img_full_path = TEST_OUTPUT_IMG_DIR + '/' + image_file
    image = cv2.imread(input_img_full_path)
    bbox, bboxes, landmarks = detector.predict(image)
    
    print('bboxes-shape---:',bboxes.shape)
    print('landmarks-shape---:',landmarks.shape)
    for bbox in bboxes:
        #print('bbox score--:',bbox[4])
        #cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
        
    for landmark in landmarks:
        #print('landmark-shape---:',landmark.shape)
        #print('landmark----:',landmark)
        
        for i in range(0, 5):
            cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
        #break
        
    cv2.imwrite(output_img_full_path, image)
    cv2.imshow('yy', image)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR:%s Input img name with .jpg \r\n" % (sys.argv[0]))
    else:
        main(sys.argv[1])
