#!/usr/bin/env python3
import argparse
import os
import sys

import tensorflow as tf
import tensorflow.contrib.slim.nets

from image_generator import ImageGenerator
from model import SGRU


def vgg_19_evaluate(image):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        _, end_points = tf.contrib.slim.nets.vgg.vgg_19(image, is_training=False)
    return end_points


def build_loss_func(image_bw, images_rgb_fake, image_rgb_real):

    lambda_weights = [0.88, 0.79, 0.63, 0.51, 0.39, 1.07]

    end_points_real = vgg_19_evaluate(image_rgb_real)

    layers = [
        'input',
        'vgg_19/conv1/conv1_2',
        'vgg_19/conv2/conv2_2',
        'vgg_19/conv3/conv3_2',
        'vgg_19/conv4/conv4_2',
        'vgg_19/conv5/conv5_2',
    ]

    losses = []

    # Iterate through Unet output collection
    collection_size = 9
    for i in range(collection_size):

        loss = tf.Variable(0.0)

        print(f'Building loss for collection image {i}')

        image_rgb_fake = images_rgb_fake[:, :, :, i*3:(i+1)*3]

        end_points_fake = vgg_19_evaluate(image_rgb_fake)

        for weight, layer in zip(lambda_weights, layers):

            if layer == 'input':
                act_fake = image_rgb_fake[0]
                act_real = image_rgb_real[0]
            else:
                act_fake = end_points_fake[layer][0]
                act_real = end_points_real[layer][0]

            # Resize image (and convert it to greyscale?)
            mask = tf.image.resize_images(image_bw[0], tf.shape(act_fake)[:2])
            mask = tf.reduce_mean(mask, axis=2)

            for filter_num in range(act_fake.shape[-1]):

                filter_fake = act_fake[:, :, filter_num]
                filter_real = act_real[:, :, filter_num]

                loss_inner = weight * tf.norm(tf.multiply(mask, filter_fake-filter_real), 1)
                loss = loss + loss_inner

        losses.append(loss)

    return tf.reduce_min(losses)


def train(loss_func, image_bw, image_rgb_real, data_dir, vgg_fname, epochs, batch_size):
    
    # Load VGG variables
    variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    vgg_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(vgg_fname, variables_to_restore,
                                                                 ignore_missing_vars=True)

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.initializers.global_variables())
        # Initialize VGG variables (these were reset during global initialization)
        vgg_init_fn(sess)

        losses = []

        for epoch in range(epochs):

            image_generator = ImageGenerator(data_dir, batch_size)

            for batch_bw, batch_rgb in image_generator.load_batches():

                feed_dict = {
                    image_bw: batch_bw,
                    image_rgb_real: batch_rgb
                }
                loss = sess.run([loss_func], feed_dict=feed_dict)
                print(f'[Epoch {epoch}, loss: {loss}')

                losses.append(loss)


def main(args):

    if not os.path.isfile(args.vgg_fname):
        sys.exit('Download VGG19 checkpoint from ' +
                 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz')

    image_rgb_real = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_real')
    image_bw = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_fake')
    model = SGRU(image_bw)
    image_rgb_fake = model.output

    loss_func = build_loss_func(image_bw, image_rgb_fake, image_rgb_real)

    train(loss_func, image_bw, image_rgb_real, args.data_dir, args.vgg_fname, args.epochs,
          args.batch_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('vgg_fname', help='VGG checkpoint filename')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training models')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
