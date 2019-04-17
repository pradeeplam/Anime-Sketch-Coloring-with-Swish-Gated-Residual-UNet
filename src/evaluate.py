#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from model import SGRU


def load_image(image_fname):
    """Load image and pad it to make dimensions divisible by 32"""
    image = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    rows, cols = image.shape[:2]
    dim_larger = max(rows, cols)
    dim_larger_pad = ((dim_larger // 32) + 1) * 32
    image_padded = np.zeros((dim_larger_pad, dim_larger_pad), dtype=np.float32)
    image_padded[:rows, :cols] = image
    # Add batch and channel dimension
    image_padded = image_padded[np.newaxis, :, :, np.newaxis]
    return image_padded, (rows, cols)


def preprocess_images(image_bw, images_rgb_fake, dims_orig):

    rows, cols = dims_orig

    # Cut BW image to original size, convert to BGR
    image_bw = image_bw[0, :rows, :cols].astype(np.uint8)
    image_bw = cv2.cvtColor(image_bw, cv2.COLOR_GRAY2BGR)

    # Cut RGB images to original size
    images_rgb_fake = images_rgb_fake[:, :rows, :cols]

    # Remove clipping, convert to uint8
    images_rgb_fake = np.minimum(np.maximum(images_rgb_fake, 0.0), 255.0)
    images_rgb_fake = images_rgb_fake.astype(np.uint8)

    # Split images ([9, rows, cols, 3] to array of [rows, cols, 3]), convert RGB -> BGR
    images_rgb_fake = [cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) for image_rgb in images_rgb_fake]

    return image_bw, images_rgb_fake


def merge_images(image_bw, images_rgb_fake):
    image_row = np.hstack([image_bw] + images_rgb_fake)
    return image_row


def show_images(image_bw, images_rgb_fake):
    images_merged = merge_images(image_bw, images_rgb_fake)
    cv2.imshow('SGRU Images', images_merged)
    cv2.waitKey(0)


def save_images(image_bw, images_rgb_fake, output_dir):
    images_merged = merge_images(image_bw, images_rgb_fake)
    fname_merged = os.path.join(output_dir, 'merged.jpg')
    cv2.imwrite(fname_merged, images_merged)
    cv2.imwrite(os.path.join(output_dir, 'bw.jpg'), image_bw)

    for idx, image_rgb_fake in enumerate(images_rgb_fake):
        fname_rgb_fake = os.path.join(output_dir, 'image_{}.jpg'.format(idx))
        cv2.imwrite(fname_rgb_fake, image_rgb_fake)


def main(args):

    image_bw, dims_orig = load_image(args.image_fname)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:

        sgru_model = SGRU()
        sgru_model.load(args.ckpt_sgru_fname)

        sess.run(init_op)

        images_rgb_fake = sess.run(sgru_model.images_rgb_fake, feed_dict={
            sgru_model.image_bw: image_bw,
        })

    image_bw, images_rgb_fake = preprocess_images(image_bw, images_rgb_fake, dims_orig)

    if args.show:
        show_images(image_bw, images_rgb_fake)

    if args.output_dir:
        save_images(image_bw, images_rgb_fake, args.output_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_fname', help='Input sketch image')
    parser.add_argument('ckpt_sgru_fname', help='Pretrained SGRU checkpoint filename')
    parser.add_argument('--show', action='store_true', default=False, help='Display results')
    parser.add_argument('--output-dir', type=str,
                        help='Where to save output. If nothing provided, nothing is saved.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)