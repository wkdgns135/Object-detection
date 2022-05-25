import cv2
import os

def file_read(category):
    fileName = os.listdir("101_ObjectCategories/" + category)

    train_img = []
    test_img = []
    i = 0
    
    for name in fileName:
        if(i < 30):
            train_img.append(cv2.imread("101_ObjectCategories/"+category+"/" + name))
        else:
            test_img.append(cv2.imread("101_ObjectCategories/"+category+"/" + name))
        i+=1

    return train_img, test_img
    