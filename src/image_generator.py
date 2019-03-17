import os
import tensorflow as tf
import cv2
import numpy as np

class ImageGenerator(object):


    def __init__(self, image_dir, batch_size, is_training=False):
        if is_training:
            self.image_paths = self.get_image_paths_train(image_dir)
        else:
            self.image_paths = self.get_image_paths_eval(image_dir)

        self.batch_size = batch_size
        self.is_training = is_training 


    def get_image_paths_train(self, image_dir):

        image_bw_dir = os.path.join(image_dir, 'images_bw') 
        image_rgb_dir = os.path.join(image_dir, 'images_rgb')

        result = []
        for path_bw in os.listdir(image_bw_dir):
            # Check extensions of filename
            if path_bw.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to bw image
            path_bw_full = os.path.join(image_bw_dir, path_bw) 

            # Construct complete path to rgb image
            path_rgb = path_bw.replace('images_bw', 'images_bw')
            path_rgb_full = os.path.join(image_rgb_dir, path_rgb)

            # Validate if colorized image exists
            if not os.path.isfile(path_rgb_full):
                continue

            result.append([path_bw_full, path_rgb_full])

        return result


    def get_image_paths_eval(self, image_dir):

        result = []
        for path_bw in os.listdir(image_dir):
            # Check extensions of filename
            if path_bw.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to bw image
            path_bw_full = os.path.join(image_dir, path_bw) 

            result.append(path_bw_full)

        return result

 

    def load_batches(self):
        if self.is_training:
            rgb_paths = [p[0] for p in self.image_paths]
            bw_paths  = [p[1] for p in self.image_paths]
            dataset = tf.data.Dataset.from_tensor_slices((rgb_paths, bw_paths))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((testing, testing2))

        
        def load_image_pairs(bw_img, rgb_img):
            return load_image(bw_img), load_image(rgb_img, cv2.IMREAD_COLOR) 

        def load_image(img_path, read_mode=cv2.IMREAD_GRAYSCALE):
            img = cv2.imread(img_path.decode(), read_mode).astype(np.float32)
            img /= 255.0
            img = cv2.resize(img, (128, 128))
            return img[:, :, np.newaxis] if read_mode == cv2.IMREAD_GRAYSCALE else img


        # Repeat indefinitely
        dataset = dataset.repeat() 

        # Unform shuffle (Change for possible speedup?)
        dataset = dataset.shuffle(buffer_size=len(self.image_paths)) 
        
        # Map path to image 
        if self.is_training:
            dataset = dataset.map(
                lambda bw_img, rgb_img: tuple(tf.py_func(
            load_image_pairs, [bw_img, rgb_img], [tf.float32, tf.float32])))
        else:
            dataset = dataset.map(load_image)

        dataset = dataset.batch(self.batch_size)

        bw_img, rgb_img = dataset.make_one_shot_iterator().get_next()

        return bw_img, rgb_img