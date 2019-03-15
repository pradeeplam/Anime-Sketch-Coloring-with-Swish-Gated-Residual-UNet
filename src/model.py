"""
Builds the UNet model model as described in paper:
"Anime Sketch Colowing with Swish-Gated Residual U-Net"
"""
import tensorflow as tf


def Conv2DLReLUBase(conv_func, inputs, filters, kernel_size=2, strides=1, padding='same',
                    kernel_initializer='he_normal', alpha=0.03):
    layer = conv_func(
        inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer)
    layer = tf.nn.leaky_relu(layer, alpha=alpha)
    return layer


def Conv2DLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=tf.layers.conv2d, *args, **kwargs)


def Conv2DTransposeLReLU(*args, **kwargs):
    return Conv2DLReLUBase(conv_func=tf.layers.conv2d_transpose, *args, **kwargs)


def Swish(inputs):
    # Do we put a convolution in here?
    # conv =
    layer = tf.nn.swish(inputs)
    layer = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer)
    return layer


class SGRU(object):

    def __init__(self, inputs):
        self.inputs = inputs

        inputs, conv1 = self._swish_gated_block('SGB_1', inputs, 96, conv1x1=False)
        inputs, conv2 = self._swish_gated_block('SGB_2', inputs, 192)
        inputs, conv3 = self._swish_gated_block('SGB_3', inputs, 288)
        inputs, conv4 = self._swish_gated_block('SGB_4', inputs, 384)
        inputs, conv5 = self._swish_gated_block('SGB_5', inputs, 480)

        swish1 = Swish(conv1)
        swish2 = Swish(conv2)
        swish3 = Swish(conv3)
        swish4 = Swish(conv4)
        swish5 = Swish(conv5)

        inputs, _ = self._swish_gated_block('SGB_5_up', inputs, 512, cat=[swish5])
        inputs, _ = self._swish_gated_block('SGB_4_up', inputs, 480, cat=[swish4])
        inputs, _ = self._swish_gated_block('SGB_3_up', inputs, 384, cat=[swish3])
        inputs, _ = self._swish_gated_block('SGB_2_up', inputs, 288, cat=[swish2])
        inputs, _ = self._swish_gated_block('SGB_1_up', inputs, 192, cat=[swish1])

        conv1_1_up = Conv2DLReLU(filters=96, kernel_size=1, inputs=inputs)
        conv1_2_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_1_up)
        conv1_3_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_2_up)
        conv1_4_up = tf.layers.Conv2D(filters=27, kernel_size=1, activation=None, padding='same',
                                  kernel_initializer='he_normal')(conv1_3_up)

        self.output = conv1_4_up

    # swish_gated block takes in a input tensor and returns two objects, one of
    # which is the concat operation found in the SGB, and the other is the
    # output of the last convolutional layer (before maxpool or deconv)
    #
    # (Think of a better variable name than cat)
    # If the cat list is an empty list, we assume we are in the down part of the
    # Unet. Otherwise, we are in the up part.
    def _swish_gated_block(self, name, inputs, filters, cat=[], conv1x1=True):

        if conv1x1:
            inputs = Conv2DLReLU(filters=filters, kernel_size=1, inputs=inputs)

        conv1 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=inputs)
        conv2 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=conv1)

        if cat == []:
            sgb_op = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
        else:
            sgb_op = Conv2DTransposeLReLU(filters=filters, inputs=conv2)

        return tf.concat([sgb_op, Swish(inputs)] + cat, axis=3), conv2

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
