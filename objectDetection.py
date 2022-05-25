import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass
from fileRead import file_read

class object_detector:
    def __init__(self, category):
        self.category = category
        self.train_data, self.test_data = file_read(category)

    def show_image(self):
        cv2.imshow("test",self.train_data[1])
        cv2.waitKey(0)
    def detector(self):
        
        self.SIFT()

    def SIFT(self, img):
        # TODO 이미지 전체의 특징점 최소값 추출후 nfeatures 설정
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures = 100)
        keypoint = sift.detect(gray)
        cv2.drawKeypoints(img, keypoint, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("SIFT features", img)
        print(len(keypoint))
        return keypoint