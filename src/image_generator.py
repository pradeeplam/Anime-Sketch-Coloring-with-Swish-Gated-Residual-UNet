import os
import random

import cv2
import numpy as np


class ImageGenerator(object):


    def __init__(self, image_dir, batch_size):
        self.image_fnames = self.get_image_fnames(image_dir)
        self.batch_size = batch_size


    def get_image_fnames(self, image_dir):


        image_bw_dir = os.path.join(image_dir, 'images_bw')
        image_rgb_dir = os.path.join(image_dir, 'images_rgb')

        result = []

        for path_bw in os.listdir(image_bw_dir):

            # Check extension of filename
            if path_bw.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            path_rgb = path_bw.replace('images_bw', 'images_bw')

            path_bw_full = os.path.join(image_bw_dir, path_bw)
            path_rgb_full = os.path.join(image_rgb_dir, path_rgb)

            if not os.path.isfile(path_rgb_full):
                continue

            result.append((path_bw_full, path_rgb_full))

        return result


    def load_image(self, fname):
        image = cv2.imread(fname).astype(np.float32)
        image /= 255.0
        image = cv2.resize(image, (128, 128))
        return image


    def load_batches(self):

        random.shuffle(self.image_fnames)
        batch_bw = []
        batch_rgb = []

        for path_bw, path_rgb in self.image_fnames:

            image_bw = self.load_image(path_bw)
            image_rgb = self.load_image(path_rgb)

            batch_bw.append(image_bw)
            batch_rgb.append(image_rgb)

            if len(batch_bw) >= self.batch_size:
                yield batch_bw, batch_rgb
                batch_bw = []
                batch_rgb = []

