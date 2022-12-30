import os
import cv2
import time
import numpy as np
from pyautogui import screenshot
from final_resut_GUI import win_work


base_dir = "origin/"
dst_dir = "result/"
dst1_dir = "final_result/"

if not os.path.exists(dst1_dir):
    os.mkdir(dst1_dir.split('/')[0])

create_file = [base_dir, dst_dir, dst1_dir]
min_val = 10
min_range = 30

win_work()
time.sleep(0.2)

try:
    for file in create_file:
        if not os.path.exists(file):
            os.mkdir(file.split('/')[0])
except:
    pass

img = screenshot()
img.save('screenshot.png')
img = cv2.imread('screenshot.png', -1)

mouse_state = False

dots = []   # 記錄座標的空串列
def show_xy(event,x,y,flags,param):
    global dot1, dot2, img, img2, mouse_state
    if flags == 1:
        if event == 1:
            dot1 = [x, y]
        if event == 0:
            img2 = img.copy()
            dot2 = [x, y]
            cv2.rectangle(img2, (dot1[0], dot1[1]), (dot2[0], dot2[1]), (0, 0, 255), 2)
            cv2.imshow('show', img2)
            mouse_state = True
    if event == 4 and mouse_state:
        print('ok')
        img_ = img[dot1[1]:dot2[1], dot1[0]:dot2[0]]
        cv2.imwrite(base_dir+'output.png', img_)
        cv2.destroyAllWindows()

weidth = int(img.shape[1]/1.2)
height = int(img.shape[0]/1.2)
cv2.namedWindow('show', 0)
cv2.resizeWindow('show', weidth, height)

cv2.imshow('show', img)
cv2.setMouseCallback('show', show_xy)

cv2.waitKey(0)
cv2.destroyAllWindows()

os.remove('screenshot.png')

img = cv2.imread(base_dir+'output.png', 0)

count = 0

def clean_file():
    for file in create_file:
        for fileName in os.listdir(file):
            os.remove(file + fileName)
        os.rmdir(file.split('/')[0])

def extract_peek(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            if i - start_i >= minimun_range:
                end_i = i
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges
def cutImage(img):
    global count
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            cv2.imwrite(dst_dir + str(count) + ".png", img1)

for i in range(0, 1):
    for fileName in os.listdir(base_dir):
        img = cv2.imread(base_dir + fileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        horizontal_sum = np.sum(adaptive_threshold, axis=1)
        peek_ranges = extract_peek(horizontal_sum, min_val, min_range)
        line_seg_adaptive_threshold = np.copy(adaptive_threshold)
        for i, peek_range in enumerate(peek_ranges):
            x = 0
            y = peek_range[0]
            w = line_seg_adaptive_threshold.shape[1]
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
        vertical_peek_ranges2d = []
        for peek_range in peek_ranges:
            start_y = peek_range[0]
            end_y = peek_range[1]
            line_img = adaptive_threshold[start_y:end_y, :]
            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = extract_peek(
                vertical_sum, min_val, min_range)
            vertical_peek_ranges2d.append(vertical_peek_ranges)
        cutImage(img)

def _img_normalization(img):
    _range = np.max(img) - np.min(img)
    return ((img - np.min(img)) / _range) * 255

def adjust_img(img, num0, num1):
    size = 100 / img.shape[num0]
    img = cv2.resize(img, None, fx=size, fy=size, interpolation=cv2.INTER_AREA)
    img = _img_normalization(img)
    if np.min(img) > 60:
        img[img < 255] = 0
    img[img > 220] = 255
    img[img < 50] = 0
    pad1 = int((100 - img.shape[num1]) / 2)
    pad2 = 100 - pad1 - img.shape[num1]
    return img, pad1, pad2

def processing(img):
    if img.shape[0] >= img.shape[1]:
        img, pad1, pad2 = adjust_img(img, 0, 1)
        img = np.pad(img, ((0, 0), (pad1, pad2)), 'constant', constant_values=(255, 255))

    elif img.shape[0] < img.shape[1]:
        img, pad1, pad2 = adjust_img(img, 1, 0)
        img = np.pad(img, ((pad1, pad2), (0, 0)), 'constant', constant_values=(255, 255))

    img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((5, 5), (5, 5)), 'constant', constant_values=(255, 255))

    return img

for fileName in os.listdir(dst_dir):
    img = cv2.imread(dst_dir+fileName, 0)
    img = processing(img)
    cv2.imwrite(dst1_dir+fileName, img)

print("\nImage processing has done. ")

# https://steam.oxxostudio.tw/category/python/ai/opencv-mouse-mosaic.html
# https://blog.51cto.com/u_12630471/3704717