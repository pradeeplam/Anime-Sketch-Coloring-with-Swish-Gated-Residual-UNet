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


def process_image(fname):
    image_in = cv2.imread(fname)
    if image_in is None:
        return
    image_out = cv2.edgePreservingFilter(image_in)
    image_out, _ = cv2.pencilSketch(image_out, sigma_r=0.09, shade_factor=0.05)
    cv2.imwrite(fname, image_out)


def pool_process(fnames_all):
    pool = Pool(processes=4)
    pool.map(process_image, fnames_all)


def single_process(fnames_all):
    for idx, fname in enumerate(fnames_all):
        print('[{}/{}] Processing {}'.format(idx+1, len(fnames_all), fname))
        process_image(fname)


def main(args):
    fnames_all = get_all_fnames(args.dir_name)
    if args.pool:
       pool_process(fnames_all)
    else:
       single_process(fnames_all)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_name', help='Input directory')
    parser.add_argument('--pool', action='store_true', help='Use process pool')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
