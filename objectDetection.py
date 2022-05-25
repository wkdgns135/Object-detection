from cv2 import KeyPoint_convert
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass
from fileRead import file_read

class object_detector:
    def __init__(self, category):
        self.category = category
        self.train_img, self.test_img = file_read(category)
        self.train_data, self.test_data = [],[]

    def show_image(self):
        cv2.imshow("test",self.train_img[1])
        cv2.waitKey(0)

    def detector(self):
        self.SIFT()

    def SIFT(self):
        # TODO 이미지 전체의 특징점 최소값 추출후 nfeatures 설정
        # min feature
        min = 10000
        for i in self.train_data + self.train_data:
            gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            keypoint = sift.detect(gray)
            min = len(keypoint) if len(keypoint) < min else min
        
        sift = cv2.xfeatures2d.SIFT_create(nfeatures = min)
        for img in self.train_img:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoint = sift.detect(gray)
            self.train_data.append(KeyPoint_convert(keypoint))

        for img in self.test_img:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoint = sift.detect(gray)
            self.test_data.append(KeyPoint_convert(keypoint))