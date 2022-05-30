from cv2 import KeyPoint_convert, kmeans
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass
from fileRead import file_read
from sklearn.cluster import KMeans

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
        self.k_means()

    def SIFT(self):
        # TODO 이미지 전체의 특징점 최소값 추출후 nfeatures 설정
        # min feature
        # min = 10000
        # for i in self.train_img + self.test_img:
        #     gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        #     sift = cv2.xfeatures2d.SIFT_create()
        #     keypoint = sift.detect(gray)
        #     # min = len(keypoint) if len(keypoint) < min else min

        sift = cv2.xfeatures2d.SIFT_create()
        for img in self.train_img:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoint = sift.detect(gray)
            keypoint = KeyPoint_convert(keypoint)
            self.train_data.append(keypoint)
            # print(keypoint.shape)

        for img in self.test_img:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoint = sift.detect(gray)
            keypoint = KeyPoint_convert(keypoint)
            self.test_data.append(keypoint)
            # print(keypoint.shape)
    
    def k_means(self):
        km = KMeans()
        
        km.fit(self.train_data)
        km.predict(self.test_data)

        print(km.score)