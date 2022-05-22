import cv2
from fileRead import file_read

train_data, test_data = file_read("ant")

cv2.imshow("train", test_data[1])
cv2.waitKey(0)