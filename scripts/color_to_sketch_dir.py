#!/usr/bin/env python3

"""Converts all images in a directory, maintaining original
directory structure. This will overwrite existing images.
"""

import argparse
import os
from fnmatch import fnmatch

import cv2


def process_image(fname):
    image_in = cv2.imread(fname)
    image_out = cv2.edgePreservingFilter(image_in)
    image_out, _ = cv2.pencilSketch(image_out, sigma_r=0.09, shade_factor=0.05)
    cv2.imwrite(fname, image_out)


def main(args):

    fnames_all = []

    for path, subdirs, files in os.walk(args.dir_name):
        fnames = [fname for fname in files if fnmatch(fname, '*.png')]
        for idx, fname in enumerate(fnames):
            fname_full = os.path.join(path, fname)
            fnames_all.append(fname_full)

    for idx, fname in enumerate(fnames_all):
        print('[{}/{}] Processing {}'.format(idx+1, len(fnames_all), fname))
        process_image(fname)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_name', help='Input directory')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
