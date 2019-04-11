import os
import tensorflow as tf
import cv2
import numpy as np


class ImageGenerator(object):

    def __init__(self, image_dir, num_cpus):
        self.rgb_paths, self.bw_paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.rgb_paths)
        self.num_cpus = num_cpus


    def get_image_paths_train(self, image_dir):

        image_bw_dir = os.path.join(image_dir, 'images_bw') 
        image_rgb_dir = os.path.join(image_dir, 'images_rgb')

        rgb_paths = []
        bw_paths = []

        for path_bw in os.listdir(image_bw_dir):
            # Check extensions of filename
            if path_bw.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to bw image
            path_bw_full = os.path.join(image_bw_dir, path_bw) 

            # Construct complete path to rgb image
            path_rgb = path_bw.replace('images_bw', 'images_rgb')
            path_rgb_full = os.path.join(image_rgb_dir, path_rgb)

            # Validate if colorized image exists
            if not os.path.isfile(path_rgb_full):
                continue

            rgb_paths.append(path_rgb_full)
            bw_paths.append(path_bw_full)

        return rgb_paths, bw_paths


    def load_image(self, img_path, read_mode):

        image = cv2.imread(img_path.decode(), read_mode).astype(np.float32)

        if read_mode == cv2.IMREAD_COLOR:
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Add 3rd dimension to grayscale
            image = image[:, :, np.newaxis]

        return image


    def load_image_pairs(self, bw_img, rgb_img):
        return (self.load_image(bw_img, cv2.IMREAD_GRAYSCALE),
                self.load_image(rgb_img, cv2.IMREAD_COLOR))


    def load_images(self):

        dataset = tf.data.Dataset.from_tensor_slices((self.bw_paths, self.rgb_paths))

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Unform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.bw_paths))
        
        # Map path to image 
        dataset = dataset.map(lambda bw_img, rgb_img: tuple(tf.py_func(
            self.load_image_pairs, [bw_img, rgb_img], [tf.float32, tf.float32])),
            self.num_cpus)

        dataset = dataset.batch(1)

        bw_img, rgb_img = dataset.make_one_shot_iterator().get_next()

        return bw_img, rgb_img