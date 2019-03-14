"""
Builds the UNet model model as described in paper:
"Anime Sketch Colowing with Swish-Gated Residual U-Net"
"""
import tensorflow as tf
from tf.layers import (Conv2D, Conv2DTranspose, Cropping2D, Input, LeakyReLU,
                       MaxPooling2D, Multiply, concatenate)
from tf.nn import leaky_relu, swish
from keras.models import Model


def Conv2DLReLUBase(conv_func, inputs, filters, kernel_size=2, strides=2, padding='same',
                    kernel_initializer='he_normal', alpha=0.03):
    layer = conv_func(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer)(inputs)
    layer = leaky_relu(alpha=alpha)(layer)
    return layer


def Conv2DLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=Conv2D, *args, **kwargs)


def Conv2DTransposeLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=Conv2DTranspose, *args, **kwargs)


def build_model():

    image_channels = 3
    inputs = Input((None, None, image_channels))

    conv1_1 = Conv2DLReLU(filters=96, kernel_size=3, inputs=inputs) 
    conv1_2 = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_1)
    max_pool1 = MaxPooling2D()(conv1_2) 

    swish1_2 = swish(inputs)
    inputs2  = concatenate([max_pool1, swish1_2])

    conv2_1 = Conv2DLReLU(filters=192, kernel_size=1, inputs=inputs2) 
    conv2_2 = Conv2DLReLU(filters=192, kernel_size=3, inputs=conv2_1)
    conv2_3 = Conv2DLReLU(filters=192, kernel_size=3, inputs=conv2_2) 
    max_pool2 = MaxPooling2D()(conv2_3)

    swish2_3 = Swish(inputs=conv2_1)
    inputs3  = concatenate([max_pool2, swish2_3])

    conv3_1 = Conv2DLReLU(filters=288, kernel_size=1, inputs=inputs3) 
    conv3_2 = Conv2DLReLU(filters=288, kernel_size=3, inputs=conv3_1) 
    conv3_3 = Conv2DLReLU(filters=288, kernel_size=3, inputs=conv3_2) 
    max_pool3 = MaxPooling2D()(conv3_3)

    swish3_4 = Swish(inputs=conv3_1)
    inputs4  = concatenate([max_pool3, swish3_4])

    conv4_1 = Conv2DLReLU(filters=384, kernel_size=1, inputs=inputs4) 
    conv4_2 = Conv2DLReLU(filters=384, kernel_size=3, inputs=conv4_1) 
    conv4_3 = Conv2DLReLU(filters=384, kernel_size=3, inputs=conv4_2) 
    max_pool4 = MaxPooling2D()(conv4_3)

    swish4_5 = Swish(inputs=conv4_1)
    inputs5  = concatenate([max_pool4, swish4_5])

    conv5_1 = Conv2DLReLU(filters=480, kernel_size=1, inputs=inputs5) 
    conv5_2 = Conv2DLReLU(filters=480, kernel_size=3, inputs=conv5_1) 
    conv5_3 = Conv2DLReLU(filters=480, kernel_size=3, inputs=conv5_2) 
    max_pool5 = MaxPooling2D()(conv5_3)

    swish5_6 = Swish(inputs=conv5_1)
    inputs6  = concatenate([max_pool5, swish5_6])

    conv6_1 = Conv2DLReLU(filters=512, kernel_size=1, inputs=inputs6) 
    conv6_2 = Conv2DLReLU(filters=512, kernel_size=3, inputs=conv6_1) 
    conv6_3 = Conv2DLReLU(filters=512, kernel_size=3, inputs=conv6_2) 
    deconv6 = Conv2DTransposeLReLU(filters=512, inputs=conv6_3) 

    swish6_5 = Swish(inputs=conv6_1)
    swish5_5 = Swish(inputs=conv5_3)
    inputs5_up = concatenate([swish6_5, swish5_5, deconv6])

    conv5_1_up = Conv2DLReLU(filters=480, kernel_size=1, inputs=inputs5_up)
    conv5_2_up = Conv2DLReLU(filters=480, kernel_size=3, inputs=conv5_1_up) 
    conv5_3_up = Conv2DLReLU(filters=480, kernel_size=3, inputs=conv5_2_up) 
    deconv5 = Conv2DTransposeLReLU(filters=480, inputs=conv5_3_up) 

    swish5_4 = Swish(inputs=conv5_1_up)
    swish4_4 = Swish(inputs=conv4_3)
    inputs4_up = concatenate([swish5_4, swish4_4, deconv5])

    conv4_1_up = Conv2DLReLU(filters=384, kernel_size=1, inputs=inputs4_up) 
    conv4_2_up = Conv2DLReLU(filters=384, kernel_size=3, inputs=conv4_1_up) 
    conv4_3_up = Conv2DLReLU(filters=384, kernel_size=3, inputs=conv4_2_up) 
    deconv4 = Conv2DTransposeLReLU(filters=384, inputs=conv4_3_up) 

    swish4_3 = Swish(inputs=conv4_1_up)
    swish3_3 = Swish(inputs=conv3_3)
    inputs3_up = concatenate([swish4_3, swish3_3, deconv4])

    conv3_1_up = Conv2DLReLU(filters=288, kernel_size=1, inputs=inputs3_up) 
    conv3_2_up = Conv2DLReLU(filters=288, kernel_size=3, inputs=conv3_1_up) 
    conv3_3_up = Conv2DLReLU(filters=288, kernel_size=3, inputs=conv3_2_up) 
    deconv3 = Conv2DTransposeLReLU(filters=288, inputs=conv3_3_up) 

    swish3_2 = Swish(inputs=conv3_1_up) 
    swish2_2 = Swish(inputs=conv2_3)
    inputs2_up = concatenate([swish3_2, swish2_2, deconv3])

    conv2_1_up = Conv2DLReLU(filters=192, kernel_size=1, inputs=inputs2_up) 
    conv2_2_up = Conv2DLReLU(filters=192, kernel_size=3, inputs=conv2_1_up) 
    conv2_3_up = Conv2DLReLU(filters=192, kernel_size=3, inputs=conv2_2_up) 
    deconv2 = Conv2DTransposeLReLU(filters=192, inputs=conv2_3_up) 

    swish2_1 = Swish(inputs=conv2_1_up)
    swish1_1 = Swish(inputs=conv1_2)
    inputs1_up = concatenate([swish2_1, swish1_1, deconv2])

    conv1_1_up = Conv2DLReLU(filters=96, kernel_size=1, inputs=inputs1_up)
    conv1_2_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_1_up) 
    conv1_3_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_2_up) 
    conv1_4_up = Conv2D(filters=27, kernel_size=1, activation=None, padding='same',
                        kernel_initializer='he_normal')(conv1_3_up)

    return Model(inputs, conv1_4_up)
