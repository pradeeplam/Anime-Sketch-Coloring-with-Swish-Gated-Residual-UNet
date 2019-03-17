#!/usr/bin/env python3
import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets

from image_generator import ImageGenerator
from model import SGRU


def vgg_19_evaluate(image):
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        _, end_points = tf.contrib.slim.nets.vgg.vgg_19(image, is_training=False)
    return end_points


def build_loss_func(image_bw, images_rgb_fake, image_rgb_real, batch_size):

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

        image_rgb_fake = images_rgb_fake[:, :, :, i*3:(i+1)*3]

        end_points_fake = vgg_19_evaluate(image_rgb_fake)

        for weight, layer in zip(lambda_weights, layers):

            if layer == 'input':
                act_fake = image_rgb_fake
                act_real = image_rgb_real
            else:
                act_fake = end_points_fake[layer]
                act_real = end_points_real[layer]

            # Resize image
            mask = tf.image.resize_images(image_bw[0], tf.shape(act_fake)[1:3])

            loss_inner = tf.norm(tf.multiply(mask, act_fake-act_real), 1)
            loss = loss + loss_inner

        losses.append(loss)

    return tf.reduce_min(losses)


def save_images(output_dir, image_rgb_fake, iteration):
    """Tile images"""
    batches = (image_rgb_fake * 255).astype(np.uint8)

    row_images = []
    for batch in batches:
        row = [batch[:, :, i*3:(i+1)*3] for i in range(9)]
        row_image = np.hstack(row)
        row_images.append(row_image)
    out_image = np.vstack(row_images)

    output_fname = os.path.join(output_dir, '{}.jpg'.format(iteration))
    cv2.imwrite(output_fname, out_image)


def train(loss_func, optim_func, image_bw, image_rgb_fake, image_rgb_real, data_dir, model_ckpt,
          vgg_ckpt, epochs, batch_size, output_dir, save_every):
    
    # Load VGG variables
    variables_to_restore = tf.contrib.framework.get_variables_to_restore()
    vgg_init_fn = tf.contrib.framework.assign_from_checkpoint_fn(vgg_ckpt, variables_to_restore,
                                                                 ignore_missing_vars=True)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(init_op)
        # Initialize VGG variables (these were reset during global initialization)
        vgg_init_fn(sess)

        losses = []

        iteration = 0

        for epoch in range(epochs):

            image_generator = ImageGenerator(data_dir, batch_size, is_training=True)

            bw_img, rgb_img = image_generator.load_batches()

            while True:
                try:
                    batch_bw, batch_rgb = sess.run([bw_img, rgb_img])
                except:
                    break
       
                feed_dict = {
                    image_bw: batch_bw,
                    image_rgb_real: batch_rgb
                }
                image_rgb_fake_out, loss, _ = sess.run([image_rgb_fake, loss_func, optim_func],
                                                        feed_dict=feed_dict)

                print('Epoch {}, iteration: {}, loss: {}'.format(epoch, iteration, loss))

                losses.append(loss)

                iteration += 1
                if iteration % save_every == 0:
                    save_images(output_dir, image_rgb_fake_out, iteration)
                    saver.save(sess, model_ckpt)


def main(args):

    if not os.path.isfile(args.vgg_ckpt):
        sys.exit('Download VGG19 checkpoint from ' +
                 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz')

    image_rgb_real = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_real')
    image_bw = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='img_fake')
    model = SGRU(image_bw)
    image_rgb_fake = model.output

    loss_func = build_loss_func(image_bw, image_rgb_fake, image_rgb_real, args.batch_size)
    optimizer_func = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss_func)

    train(loss_func, optimizer_func, image_bw, image_rgb_fake, image_rgb_real, args.data_dir,
          args.model_ckpt, args.vgg_ckpt, args.epochs, args.batch_size, args.output_dir,
          args.save_every)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('model_ckpt', help='This network\'s checkpoint file')
    parser.add_argument('vgg_ckpt', help='VGG checkpoint filename')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training models')
    parser.add_argument('--save-every', type=int, default=1, help='Save image every n iterations')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
