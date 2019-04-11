#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

from image_generator import ImageGenerator
from model import SGRU


def vgg_19_evaluate(image):
    vgg_mean = np.array([123.68, 116.779, 103.939])
    image_normalized = image - vgg_mean
    with slim.arg_scope(tf.contrib.slim.nets.vgg.vgg_arg_scope()):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            _, end_points = tf.contrib.slim.nets.vgg.vgg_19(image_normalized, is_training=False)
    return end_points


def build_loss_func(sgru_model, image_rgb_real):

    image_bw = sgru_model.image_bw
    images_rgb_fake = sgru_model.images_rgb_fake

    layers_weights = [
        (0.88, 'input'),
        (0.79, 'vgg_19/conv1/conv1_2'),
        (0.63, 'vgg_19/conv2/conv2_2'),
        (0.51, 'vgg_19/conv3/conv3_2'),
        (0.39, 'vgg_19/conv4/conv4_2'),
        (1.07, 'vgg_19/conv5/conv5_2')
    ]

    end_points_real = vgg_19_evaluate(image_rgb_real)
    end_points_fake = vgg_19_evaluate(images_rgb_fake)

    losses_layers = []

    for weight, layer in layers_weights:

        if layer == 'input':
            act_fake = images_rgb_fake
            act_real = image_rgb_real
        else:
            act_fake = end_points_fake[layer]
            act_real = end_points_real[layer]

        mask = tf.image.resize_images(image_bw, tf.shape(act_fake)[1:3])
        # mask.shape = [1, rows, cols, 1]

        loss_layer = tf.abs(act_fake - act_real)
        # loss_layer.shape = [9, rows, cols, 3]

        loss_layer = tf.reduce_mean(loss_layer, reduction_indices=[3])
        # loss_layer.shape = [9, rows, cols]

        loss_layer = tf.expand_dims(loss_layer, -1)
        # loss_layer.shape = [9, rows, cols, 1]

        loss_layer = mask * loss_layer
        # loss_layer.shape = [9, rows, cols, 1]

        loss_layer = weight * tf.reduce_mean(loss_layer, reduction_indices=[1, 2])
        # loss_layer.shape = [9, 1]

        losses_layers.append(loss_layer)

    loss_sum = sum(losses_layers)
    loss_min = tf.reduce_sum(tf.reduce_min(loss_sum, reduction_indices=0))
    loss_mean = tf.reduce_sum(tf.reduce_mean(loss_sum, reduction_indices=0))
    loss = loss_min * 0.999 + loss_mean * 0.001

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Loss Min', loss_min)
    tf.summary.scalar('Loss Mean', loss_mean)

    return loss


def save_images(output_fname, images_rgb_fake, image_rgb_real, image_bw):
    """Tile images"""

    # Remove clipping
    images_rgb_fake = np.minimum(np.maximum(images_rgb_fake, 0.0), 255.0)

    rgb_fake = images_rgb_fake.astype(np.uint8)
    rgb_real = image_rgb_real.astype(np.uint8)
    image_bw = image_bw.astype(np.uint8)

    # image_bw.shape, rgb_real.shape, rgb_fake.shape
    # (1, 128, 128, 1) (1, 128, 128, 3) (9, 128, 128, 3)

    rgb_images = [rgb_real[0]] + [rgb_image for rgb_image in rgb_fake]
    out_images = [cv2.cvtColor(image_bw[0], cv2.COLOR_GRAY2BGR)]
    out_images += [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in rgb_images]
    out_image = np.hstack(out_images)
    cv2.imwrite(output_fname, out_image)


def train(sgru_model, loss_func, optim_func, image_rgb_real, args):

    # Join the log directory with the experiment name
    output_dir = os.path.join(args.output_dir, args.name)
    vgg_ckpt = os.path.join(args.data_dir, 'vgg_19.ckpt')

    # Load VGG variables
    vgg_init_fn = slim.assign_from_checkpoint_fn(vgg_ckpt, slim.get_model_variables('vgg_19'))

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Tell tensorflow not to hog all gpu memory and to multithread
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=args.num_cpus,
                               intra_op_parallelism_threads=args.num_cpus)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:

        # Summary operations for tensorboard
        summary_op = tf.summary.merge_all()
        writer = tf.summary.FileWriter(output_dir, graph=sess.graph)

        # Initialize all variables
        sess.run(init_op)

        # Initialize VGG variables (these were reset during global initialization)
        vgg_init_fn(sess)

        losses = []

        image_dir = os.path.join(args.data_dir, 'images')
        image_generator = ImageGenerator(image_dir, args.num_cpus)

        image_bw_op, image_rgb_op = image_generator.load_images()

        for epoch in range(args.epochs):

            for image_num in range(image_generator.num_images):

                image_bw, image_rgb = sess.run([image_bw_op, image_rgb_op])

                feed_dict = {
                    sgru_model.image_bw: image_bw,
                    image_rgb_real: image_rgb
                }
                out_list = [sgru_model.images_rgb_fake, loss_func, optim_func, summary_op]
                images_rgb_fake_out, loss, _ , summary = sess.run(out_list, feed_dict=feed_dict)

                # Report to tensorboard all the summaries at the current timestep
                writer.add_summary(summary, epoch*image_generator.num_images + image_num)

                print('Epoch {}, image number: {}, loss: {}'.format(epoch, image_num, loss))

                losses.append(loss)

                if image_num % args.save_every == 0:
                    image_dir = os.path.join(output_dir, 'images')
                    if not os.path.isdir(image_dir):
                        os.mkdir(image_dir)
                    image_fname = os.path.join(image_dir, '{}_{}.jpg'.format(epoch, image_num))
                    save_images(image_fname, images_rgb_fake_out, image_rgb, image_bw)
                    sgru_model.save(os.path.join(output_dir, 'model.ckpt'))


def main(args):

    if not os.path.isfile(os.path.join(args.data_dir, 'vgg_19.ckpt')):
        sys.exit('Download VGG19 checkpoint from ' +
                 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz ' +
                 'and save it to the root of your data_dir')

    image_rgb_real = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='img_real')
    model = SGRU(summarize=args.summarize)

    loss_func = build_loss_func(model, image_rgb_real)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    optimizer_func = optimizer.minimize(loss_func, var_list=model.params)

    train(model, loss_func, optimizer_func, image_rgb_real, args)


def timestamp():
    return datetime.now().strftime('sgru-%Y-%m-%d-%H-%M-%S-%f')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Specify learning rate')
    parser.add_argument('--save-every', type=int, default=400, help='Save image every n iterations')
    parser.add_argument('--num-cpus', type=int, default=1, help='Num CPUs to load images with')
    parser.add_argument('--summarize', action='store_true',
                        help='Summarize vars and images for Tensorboard')
    parser.add_argument('--name', type=str, default=timestamp(),
                        help='Name of the experiment (defaults to timestamp)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
