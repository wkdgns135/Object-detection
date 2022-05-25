from objectDetection import *

if __name__ =='__main__':
    category = "ant"
    detector = object_detector(category=category)
    for i in detector.train_data:
        detector.SIFT(img=i)
    detector.show_image()