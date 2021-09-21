import os
import random

import cv2
import numpy as np
from numpy.lib import math

import torch
import kornia


msra_dataset = './datasets/MSRA-TD500/'
trainset = os.path.join(msra_dataset, 'train')


def custom_box_points(x, y, w, h, img_w, img_h, angle):
    l, t, b, r = x, y, img_h - (y + h), img_w - (x + w)
    m = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    m_dash = []
    for i in range(4):
        tx = l * np.cos(angle) - t * np.sin(angle) - m[i][0]
        ty = t * np.cos(angle) + l * np.sin(angle) - m[i][1]
        m_dash.append((tx, ty))
    return np.array(m_dash)


def affine_transform():
    pass


if __name__ == "__main__":
    images = list(filter(lambda x: x.lower().endswith('.jpg'), os.listdir(trainset)))
    img_path = os.path.join(trainset, random.choice(images))
    # img_path = os.path.join(trainset, images[0])
    gt_path = os.path.splitext(img_path)[0] + '.gt'
    # print(gt_path)
    # print(img_path)
    img = cv2.imread(img_path)
    print('Image size', img.shape)
    mask = np.zeros_like(img)
    with open(gt_path, 'r') as f:
        gt_lines = f.readlines()
    for line in gt_lines:
        idx, diffl, x, y, w, h, angle = map(float, line.strip().split(' '))
        rect = ((x + w/2, y + h/2), (w, h), angle * 180 / math.pi)
        box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
        # box = custom_box_points(x, y, w, h, img.shape[1], img.shape[0], angle)
        box = np.int0(box)
        print(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
        # cv2.rectangle(img, x, y, x+w, y+h)    
        cv2.imwrite('out.png', img)
        cv2.imwrite('mask.png', mask)
    pass
