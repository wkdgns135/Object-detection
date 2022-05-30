from objectDetection import *

if __name__ =='__main__':
    category = "ant"
    detector = object_detector(category=category)
    detector.detector()

