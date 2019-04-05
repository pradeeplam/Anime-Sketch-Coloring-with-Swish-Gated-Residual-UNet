#!/usr/bin/env python3

"""Converts all images in a directory, maintaining original
directory structure. This will overwrite existing images.
"""
import argparse
import os
from fnmatch import fnmatch
from multiprocessing import Pool
import img_utils as iu
from keras.models import load_model
import numpy as np
import sys

import cv2


def get_all_fnames(dir_name):
    fnames_all = []
    for path, subdirs, files in os.walk(dir_name):
        fnames = []
        for fname in files:
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                if fnmatch(fname, ext):
                    fnames.append(fname)
        for idx, fname in enumerate(fnames):
            fname_full = os.path.join(path, fname)
            fnames_all.append(fname_full)
    return fnames_all


def process_image_sketch(fname):
    image_in = cv2.imread(fname)
    if image_in is None:
        return

    if not os.path.isfile('mod.h5'):
        sys.exit('Sketch model file missing! Download sketchKeras model from ' +
                 'https://github.com/lllyasviel/sketchKeras/releases ' +
                 'and save it to the scripts folder')

    mod = load_model('mod.h5')
    width = float(image_in.shape[1])
    height = float(image_in.shape[0])

    image_in = image_in.transpose((2, 0, 1))
    light_map = np.zeros(image_in.shape, dtype=np.float)

    for channel in range(3):
        light_map[channel] = iu.get_light_map(image_in[channel])

    light_map = iu.normalize_img(light_map)
    light_map = iu.add_rgb_channel(light_map)
    edge_pred = mod.predict(light_map, batch_size=1)
    edge_pred = edge_pred.transpose((3, 1, 2, 0))[0]
    #color_sketch = iu.get_color_sketch(edge_pred)
    sketch = np.amax(edge_pred, 2)
    #enhanced_sketch = iu.get_enhanced_sketch(image_out)
    #pured_sketch = iu.get_pured_sketch(image_out)
    sketch = iu.get_sketch(sketch)
    cv2.imwrite(fname, sketch)



def process_image_resize(fname, new_size=(512, 512)):
    image_in = cv2.imread(fname)
    if image_in is None:
        return

    rows, cols, _ = image_in.shape

    if rows == new_size[0] and cols == new_size[1]:
        return
    if rows > cols:
        pad = (rows - cols) // 2
        if pad > 0:
            image_in = image_in[pad:-pad, :, :]
    elif cols > rows:
        pad = (cols - rows) // 2
        if pad > 0:
            image_in = image_in[:, pad:-pad, :]
    image_out = cv2.resize(image_in, new_size)

    #new resize
    # if (cols > rows):
    #     image_in = cv2.resize(image_in, (512, int(512 / cols * rows)), interpolation=cv2.INTER_AREA)
    #     new_cols = new_size[1]
    #     new_rows = int(new_width / cols * rows)
    # else:
    #     image_in = cv2.resize(image_in, (int(512 / rows * cols), 512), interpolation=cv2.INTER_AREA)
    #     new_rows = new_size[0]
    #     new_cols = int(new_rows / rows * cols)
    # image_out = image_in[0:int(new_rows), 0:int(new_cols), :]
    
    cv2.imwrite(fname,image_out)


def process_image_remove(fname):
    """Removes bad images from images_rgb or images_bw"""

    bw_path = fname.replace('images_rgb', 'images_bw')
    rgb_path = fname.replace('images_bw', 'images_rgb')
    image_bw = cv2.imread(bw_path)
    image_rgb = cv2.imread(rgb_path)

    if image_bw is None or image_rgb is None:
        for path in [path_bw, path_rgb]:
            if os.path.isfile(path):
                print(f'Removing {path}')
                os.remove(path)


def pool_process(fnames_all, process_image):
    pool = Pool(processes=4)
    pool.map(process_image, fnames_all)


def single_process(fnames_all, process_image):
    for idx, fname in enumerate(fnames_all):
        if idx % 200 == 0:
            print('[{}/{}] Processing {}'.format(idx+1, len(fnames_all), fname))
        process_image(fname)


def main(args):
    fnames_all = get_all_fnames(args.dir_name)

    process_image_func = {
        'resize': process_image_resize,
        'sketch': process_image_sketch,
        'remove': process_image_remove
    }[args.process_type]

    if args.pool:
       pool_process(fnames_all, process_image_func)
    else:
       single_process(fnames_all, process_image_func)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_name',
                        help='Input directory')
    parser.add_argument('process_type',
                        choices=['resize', 'sketch', 'remove'],
                        help='Type of processing')
    parser.add_argument('--pool',
                        action='store_true',
                        help='Use process pool')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
