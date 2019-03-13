#!/usr/bin/env python3

import argparse

from keras.models import load_model

from model import build_model_gen, build_model_dis


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


def train(model_g, model_d, generators):

    """Note to project teammates!!!
    
    This is just some sample code as a placeholder that I grabbed from here:
    https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

    I'm not sure if this is how we are going to combine the models,
    it depends on the model implementation

    """
    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    model_d.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])

    # Input images and their conditioning images
    image_a = Input(img_shape)
    image_b = Input(img_shape)

    # By conditioning on B generate a fake version of A
    fake_a = model_g(image_b)

    # For the combined model we will only train the generator
    model_d.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = model_d([fake_a, image_b])

    model_combined = Model(inputs=[image_a, image_b], outputs=[valid, fake_a])
    model_combined.compile(loss=['mse', 'mae'],
                           loss_weights=[1, 100],
                           optimizer=optimizer)

def main(args):

    if args.resume:
        model_g = load_model(args.model_gen_fname)
        model_d = load_model(args.model_dis_fname)
    else:
        model_g = build_model_gen()
        model_d = build_model_dis()

    generators = build_generators(args.data_dir)

    train(model_g, model_d, generators)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Directory containing image subdirs')
    parser.add_argument('model_gen_fname', help='Generator model filename')
    parser.add_argument('model_dis_fname', help='Discriminator model filename')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training models')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
