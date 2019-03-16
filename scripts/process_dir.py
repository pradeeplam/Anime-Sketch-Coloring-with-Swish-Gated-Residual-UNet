#!/usr/bin/env python3

"""Converts all images in a directory, maintaining original
directory structure. This will overwrite existing images.
"""
import argparse
import os
from fnmatch import fnmatch
from multiprocessing import Pool

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

    image_out = cv2.edgePreservingFilter(image_in)
    image_out, _ = cv2.pencilSketch(image_out, sigma_r=0.09, shade_factor=0.05)
    cv2.imwrite(fname, image_out)


def process_image_resize(fname, new_size=(224, 224)):

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
    cv2.imwrite(fname, image_out)


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
