import cv2
from scipy import ndimage
import numpy as np

def get_light_map(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1,2,0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = gray.reshape((gray.shape[0],gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 128.0
    return highPass

def normalize_img(img):
    return img / np.max(img)


def add_rgb_channel(img):
    zeros = np.zeros((1,3,512,512), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0))

def get_color_sketch(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    return mat

def get_sketch(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat

def get_pured_sketch(img):
    mat = img.astype(np.float)
    mat[mat<0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat

def get_enhanced_sketch(img):
    mat = img.astype(np.float)
    mat[mat<0.1] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat
