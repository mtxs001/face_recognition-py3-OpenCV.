"""
作者：漫天星沙

一些函数
"""



import cv2
import os
import numpy as np

IMAGE_SIZE=64

def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)  # 初始化图像尺寸
    h, w, _ = image.shape
    longer_edge = max(h, w)
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longer_edge:
        dh = longer_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longer_edge:
        dw = longer_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    adjusted_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[120, 120, 220])  # 函数用于给图片添加边界，就像一个相框一样的东西
    adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(adjusted_image, (height, width))


def read_path(path_name, h=IMAGE_SIZE, w=IMAGE_SIZE):
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path, h, w)
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, h, w)
                images.append(image)
                labels.append(path_name)

    return images, labels

# 为每一类数据赋予唯一的标签值

def label_id(label, users, user_num):
    for i in range(user_num):
        if label.endswith(users[i]):
            return i

# 从指定位置读数据

def load_dataset(path_name):
    users = os.listdir(path_name)  # 返回指定的文件夹包含的文件或文件夹的名字的列表
    user_num = len(users)
    Image, Label = read_path(path_name)
    images_np = np.array(Image)
    labels_np = np.array([label_id(label, users, user_num) for label in Label])
    return images_np, labels_np,user_num


images=[]
labels=[]
