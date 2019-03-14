import os
import random

import tensorflow as tf


class ImageGenerator(object):


    def __init__(self, image_dir, batch_size):
        self.image_fnames = self.get_image_fnames(image_dir)
        self.batch_size = batch_size


    def get_image_fnames(self, image_dir):

        image_bw_dir = os.path.join(args.data_dir, 'images_bw')
        image_rgb_dir = os.path.join(args.data_dir, 'images_rgb')

        result = []

        for path_bw in os.listdir(image_bw_dir):

            # Check extension of filename
            if path_bw.split('.')[-1] not in ['.jpg', '.jpeg', '.png', '.gif']:
                continue

            # Check that RGB image exists
            path_rgb = path_bw.replace('images_bw', 'images_bw')
            if not os.path.isfile(path_rgb):
                continue

            result.append((path_bw, path_rgb))

        return result


    def load_image(self, fname):
        image = tf.read_file(fname)
        image = tf.image.decode_image(image)
        image /= 255.0
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

            if len(batch) >= self.batch_size:
                yield batch_bw, batch_rgb
                batch_bw = []
                batch_rgb = []

