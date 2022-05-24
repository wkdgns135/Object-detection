import cv2
from fileRead import file_read

class object_detector:
    def __init__(self, category):
        self.category = category
        self.train_data, self.test_data = file_read(category)
    def show_image(self):
        cv2.imshow("test",self.train_data[1])
        cv2.waitKey(0)