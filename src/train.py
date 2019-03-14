#!/usr/bin/env python3

import argparse

from image_generator import ImageGenerator
from model import build_model



def build_loss_func(image_bw, image_rgb_fake, image_rgb_real, model):

    vgg19 = tf.contrib.slim.nets.vgg.vgg19

    lambda_weights = [0.88, 0.79, 0.63, 0.51, 0.39, 1.07]

    _, end_points_real = nets.vgg.vgg_19(image_rgb_real)
    _, end_points_fake = nets.vgg.vgg_19(image_rgb_fake)

    layers = [
        'input',
        'vgg_19/conv1/conv1_2',
        'vgg_19/conv2/conv2_2',
        'vgg_19/conv3/conv3_2',
        'vgg_19/conv4/conv4_2',
        'vgg_19/conv5/conv5_2',
    ]

    losses = tf.zeros(9, trainable=False)

    # Iterate through Unet output collection
    for i in range(9):

        for weight, layer in zip(lambda_weights, layers):

            # Grab ith image in Unet output collection
            act_fake = end_points_fake[layer][i*3:(i+1)*3]
            act_real = end_points_real[layer]

            mask = tf.image.resize(image_bw, act_real.shape)

            for filter_num in range(act_real.shape[-1]):

                filter_fake = act_fake[:, :, filter_num]
                filter_real = act_real[:, :, filter_num]

                loss_inner = weight * tf.norm(tf.mul(mask, filter_fake-fake_real), 1)
                tf.assign_add(losses[i], loss_inner)

    loss = tf.reduce_min(losses)
    return loss


def train(model, loss_func, image_bw, image_rgb_real, data_gen, saver, epochs, batch_size):
    
    with tf.Session() as sess:

        losses = []

        for epoch in range(epochs):

            image_generator = ImageGenerator(args.data_dir, args.batch_size)

            for batch_bw, batch_rgb in image_generator.load_batches():

                feed_dict = {
                    image_bw: batch_bw,
                    image_rgb_real: batch_rgb
                }
                loss = sess.run([loss_func], feed_dict=feed_dict)

            losses.append(loss)


def main(args):

    saver = tf.train.Saver()

    image_rgb_real = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_real')
    image_bw = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='img_fake')
    model = build_model(image_bw)
    image_rgb_fake = model(image_bw)

    loss_func = build_loss_func(image_bw, image_rgb_fake, image_rgb_real, model)

    train(model, loss_func, image_bw, image_rgb_real, image_generator, saver, args.epochs,
          args.batch_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('model_fname', help='Model filename')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--resume', action='store_true', help='Resume training models')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
