#!/usr/bin/env python3

import argparse

import cv2
import numpy as np


def main(args):

    image_in = cv2.imread(args.fname_input)
    image_out = cv2.edgePreservingFilter(image_in)
    image_out, _ = cv2.pencilSketch(image_out, sigma_r=0.09, shade_factor=0.05)

    if args.visualize:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)
        image_both = np.hstack((image_in, image_out))
        cv2.imshow('Pencil sketch', image_both)
        cv2.waitKey(0)

    cv2.imwrite(args.fname_output, image_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname_input', help='Input filename')
    parser.add_argument('fname_output', help='Output filename')
    parser.add_argument('-v', '--visualize', action='store_true', help='Show image')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
