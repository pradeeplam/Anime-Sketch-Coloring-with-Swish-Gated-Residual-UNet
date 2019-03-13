#!/usr/bin/env python3

import argparse

from keras.models import load_model

from model import build_model


def build_generators(data_dir):

    """From https://keras.io/preprocessing/image/"""

    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    image_bw_generator = image_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'images_bw'),
        class_mode=None,
        seed=seed)

    image_rgb_generator = mask_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'images_rgb'),
        class_mode=None,
        seed=seed)

    # Combine generators into one which yields BW and RGB images
    train_generator = zip(image_generator, mask_generator)

    return image_bw_generator, image_rgb_generator


def train(model, generators):
    pass


def main(args):

    if args.resume:
        model = load_model(args.model_fname)
    else:
        model = build_model()

    generators = build_generators(args.data_dir)

    train(model, generators)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('model_fname', help='Model filename')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training models')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
