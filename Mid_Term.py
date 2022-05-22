#TODO 이미지 플립
import cv2
fname ='../Data/Lena.png'
img = cv2.imread(fname)
assert img is not None # Check if image wassuccessfully read. print('read {}'.format(fname))
print('shape:', img.shape)
print('dtype:', img.dtype)
img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)# 흑백 영상
assert img is not None
print('read {} asgrayscale'.format(fname.path))
print('shape:', img.shape)
print('dtype:', img.dtype)

fname = "Lena.png" ; img = cv2.imread(fname)
assert img is not None
print("original image shape:", img.shape)
cv2.imshow("img", img)

# 영상 사이즈 조절 방법 1: width, height를 정수로 명시
w, h = 200, 256
res_img = cv2.resize(img, (w, h)) # opencv.org 참고
print("res_img shape", res_img.shape)
#cv2.imshow("res_img", res_img); cv2.waitKey(0)

# 영상 사이즈 조절 방법 2: 줄이는 비율을 사용하는 방법
wf, hf = 0.5, 1.5
re_img = cv2.resize(img, (0,0), None, wf, hf)

# 뒤집기: flip
imgflipx = cv2.flip(img, 0) # x축을 중심으로 뒤집기
cv2.imshow("img flip around x-axis", imgflipx)
imgflipx = cv2.flip(img, 1) # y축을 중심으로 뒤집기
cv2.imshow("img flip around y-axis", imgflipx)
imgflipx = cv2.flip(img, -1) # x축, y축 모두 중심으로 뒤집기
cv2.imshow("img flip around x, y-axis", imgflipx)
cv2.waitKey(0)
outfname = "img_xy.png" # "img_xy.jpg"
cv2.imwrite(outfname, imgflipx)

#TODO 영상에 마우스 왼쪽 버튼 클릭시 빨간색 원을 그리는 프로그램
import cv2, numpy as np
fname = "Lena.png"; img = cv2.imread(fname)
image_to_show = np.copy(img)
# 마우스 콜백함수 정의
def mouse_callback(event, x, y, flags, param):
    global image_to_show
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image_to_show, (x, y), 30, (0, 0, 255), cv2.FILLED)
# 창과 마우스콜백함수 연결
cv2.namedWindow("img")
cv2.setMouseCallback("img", mouse_callback)
while True:
    cv2.imshow("img", image_to_show)
    k = cv2.waitKey(10)
    if k == 27:
        break
cv2.destroyAllWindows()

#TODO 영상 블러링
import cv2
import numpy as np

img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

kernel_identity = np.array([[0,0,0],[0,1,0],[0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32) / 9.0
kernel_5x5 = np.ones((5,5), np.float32) / 25.0

cv2.imshow('Original', img)

output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('identity filter', output)

output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3 filter', output)

output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x3 filter', output)

cv2.waitKey(0)

#TODO 영상 모션블러
import cv2
import numpy as np

img = cv2.imread('images/input.jpg')
cv2.imshow('Original', img)

size = 15

kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

output = cv2.filter2D(img, -1, kernel_motion_blur)

cv2.imshow('Motion blur', output)
cv2.waitKey(0)

#TODO 영상 샤프닝
import cv2, numpy as np
img = cv2.imread('images/input.jpg')
cv2.imshow('Original', img)

kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9, -1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1],[1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1-1],[-1,2,2,2,-1],[-1,2,8,2,-1][-1,2,2,2-1],[-1,-1,-1,-1,-1]]) / 8.0

output = cv2.filter2D(img, -1, kernel_sharpen_1)
cv2.imshow('Sharpening', output)

output = cv2.filter2D(img, -1, kernel_sharpen_2)
cv2.imshow('Excessive sharpening', output)

output = cv2.filter2D(img, -1, kernel_sharpen_3)
cv2.imshow('Edge sharpening', output)
cv2.waitKey(0)

#TODO 영상 엠보싱
import cv2, numpy as np

img_emboss_input = cv2.imread('images/input.jpg')
cv2.imshow('Original', img_emboss_input)

kernel_emboss_1 = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],[-1,0,1],[0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],[0,0,0],[0,0,-1]])

img_gray = cv2.cvtColor(img_emboss_input, cv2.COLOR_BGR2GRAY)

output = cv2.filter2D(img_gray, -1, kernel_emboss_1) + 128
cv2.imshow('Embossing - SW', output)

output = cv2.filter2D(img_gray, -1, kernel_emboss_2) + 128
cv2.imshow('Embossing - SE', output)

output = cv2.filter2D(img_gray, -1, kernel_emboss_3) + 128
cv2.imshow('Embossing - NW', output)
cv2.waitKey(0)

#TODO 에지 탐지
import cv2, numpy as np

img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# Sobel
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Canny
canny = cv2.Canny(img, 50, 240)

cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.waitKey(0)

#TODO 침식과 팽창
import cv2, numpy as np

img = cv2.imread('input.jpg', 0)

kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
img_dilation = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)

#TODO 비네트
import cv2, numpy as np

img = cv2.imread('input.jpg')
rows, cols = img.shape[:2]

kernel_x = cv2.getGaussianKernel(cols, 200)
kernel_y = cv2.getGaussianKernel(rows, 200)
kernel = kernel_y * kernel_x.T
mask = 255 * kernel / np.linalg.norm(kernel)
output = np.copy(img)

for i in range(3):
    output[:,:,i] = output[:,:,i] * mask

cv2.imshow('Original', img)
cv2.imshow('Vignette', output)
cv2.waitKey(0)

#TODO 영상 대비 향상 Histogram Equalization for grayscale image
import cv2, numpy as np
img = cv2.imread('input.jpg', 0)

histeq = cv2.equalizeHist(img)
cv2.imshow('Original', img)
cv2.imshow('Histogram equalized', histeq)
cv2.waitKey(0)

#TODO 영상 대비 향상 Histogram Equalization for Color image
import  cv2, numpy as np
img = cv2.imread('input.jpg')

img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', output)
cv2.waitKey(0)

#TODO 웹캠에 접근
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

#TODO 키보드 입력
import cv2
def print_howto():
    print("""Chage color space of the input video stream using keyboard controls. The control keys are:\n
    \t1. Grayscale - press 'g'\n
    \t2. YUV - press 'y'\n
    \t3. HSV - press 'h'\n
    \t3. GBR - press 'r'""")

if __name__ == '__main__':
    print_howto()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cur_mode = None
    pre_frame = None
    flag = False
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        output = frame
        c = cv2.waitKey(1)
        if c == 27:
            break

        if c == ord('g') or c == ord('y') or c == ord('h') or c == ord('r'):
            cur_mode = c
            flag = False
        else:
            if c != -1:
                flag = True

        if cur_mode == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_mode == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_mode == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif cur_mode == ord('r'):
            output = frame

        if flag:
            cv2.putText(output, 'Wrongg input',(50,50),0,1,(0,0,0),2,cv2.LINE_AA)

        cv2.imshow('WebCam', output)
    cap.release()
    cv2.destroyAllWindows()

#TODO 마우스 입력
import cv2, numpy as np

def detect_quadrant(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > width/2:
            if y > height/2:
                point_top_left = (int(width/2), int(height/2))
                point_bottom_right = (width - 1, height-1)
            else:
                point_top_left = (int(width/2),0)
                point_bottom_right = (width-1, int(height/2))
        else:
            if y > height/2:
                point_top_left = (0, int(height/2))
                point_bottom_right = (int(width/2), height-1)
            else:
                point_top_left = (0,0)
                point_bottom_right = (int(width/2), int(height/2))

        img = param["img"]
        cv2.rectangle(img, (0,0), (width-1, height-1), (255,255,255), -1)
        cv2.rectangle(img, point_top_left, point_bottom_right, (1,100,0),-1)

if __name__ == '__main__':
    width, height = 640, 480
    img = 255 * np.ones((height, width,3), dtype=np.uint8)
    cv2.namedWindow('Input window')
    cv2.setMouseCallback('Input window', detect_quadrant, {"img":img})

    while True:
        cv2.imshow('Input window', img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cv2.destroyAllWindows()

#TODO 비디오 스트림과 마우스 상호작용
import cv2, numpy as np

def update_pts(params, x, y):
    global x_init, y_init
    params["top_left_pt"] = (min(x_init, x), min(y_init,y))
    params["bottom_right_pt"] = (max(x_init,x), max(y_init,y))
    img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]

def draw_rectangle(event, x, y, flags, params):
    global x_init, y_init, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        update_pts(params, x , y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        update_pts(params, x, y)

if __name__ =='__main__':
    drawing = False
    event_params = {"top_left_pt": (-1,-1), "bottom_right_pt" : (-1,-1)}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError('Cannot open webcam')

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', draw_rectangle, event_params)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        (x0,y0), (x1, y1) = event_params["top_left_pt"],event_params["bottom_right_pt"]
        img[y0:y1, x0:x1]  = 255 - img[y0:y1, x0:x1]
        cv2.imshow('Webcam', img)

        c = cv2.waitKey(1)
        if c == 27:
            break;

    cap.release()
    cv2.destroyAllWindows()

#TODO 미디안 필터
import cv2, numpy as np
img = cv2.imread('input.jpg')
output = cv2.medianBlur(img, ksize=7)
cv2.imshow('Input', img)
cv2.imshow('Median filter', output)
cv2.waitKey(0)

#TODO 라플라시안 필터
edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=10)
ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

#TODO 임계치 적용
edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=10)
ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

#TODO 가우시안 필터, 양방향성 필터
import cv2, numpy as np

img = cv2.imread('input.jpg')

img_gaussian = cv2.GaussianBlur(img,(13,13), 0)
img_bilateral = cv2.bilateralFilter(img, 13, 70, 50)

cv2.imshow('Original', img)
cv2.imshow('Gaussian filter', img_gaussian)
cv2.imshow('Bilateral filter', img_bilateral)
cv2.waitKey(0)

#TODO 영상의 카툰화
import cv2, numpy as np

def print_howto():
    print("""Chage cartoonizing mode of image:\n
    \t1. Cartoonize without color - press 's'\n
    \t2. Carttonize with color - press 'c'
    """)

def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
    img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = cv2.medianBlur(img_gray, 7)

    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)

    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    dst = np.zeros(img_gray.shape)
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

if __name__ == '__main__':
    print_howto()
    cap = cv2.VideoCapture(0)

    cur_mode = None
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        c = cv2.waitKey(1)
        if c == 27:
            break
        if c!= -1 and c!= 255 and c!= cur_mode:
            cur_mode= c
        if cur_mode == ord('s'):
            cv2.imshow('Cartoonize', cartoonize_image(frame, ksize=5, sketch_mode=True))
        elif cur_mode == ord('c'):
            cv2.imshow('Cartoonize', cartoonize_image(frame, ksize=5, sketch_mode=False))
        else:
            cv2.imshow('Cartoonize', frame)
    cap.release()
    cv2.destroyAllWindows()

#TODO 얼굴 검출(탐지)과 추적
import cv2, numpy as np
cap = cv2.VideoCapture(0);
face_cascade = cv2.CascadeClassifier('./files/haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1); if c == 27: break ;
cap.release();cv2.destroyAllWindows()

#TODO 눈 검출
import cv2, numpy as np
face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_eye.xml')
if face_cascade.empty(): raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty(): raise IOError('Unable to load the eye cascade classifier xml file')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            cv2.circle(roi_color, center, radius, (0, 255, 0), 3)
    cv2.imshow('Eye Detector', frame)
    c = cv2.waitKey(1); if c == 27: break ;

cap.release();cv2.destroyAllWindows()

#TODO 코 검출
import cv2, numpy as np
nose_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_nose.xml')
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    nose_rects= nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break
    cv2.imshow('Nose Detector', frame)
    c = cv2.waitKey(1); if c == 27: break ;

cap.release();cv2.destroyAllWindows()

#TODO 입 검출
import cv2, numpy as np
mouth_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects= mouth_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=11)
    for (x,y,w,h) in mouth_rects:
        y = int(y -0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break
    cv2.imshow('Mouth Detector', frame)
    c = cv2.waitKey(1); if c == 27: break;
cap.release();cv2.destroyAllWindows()