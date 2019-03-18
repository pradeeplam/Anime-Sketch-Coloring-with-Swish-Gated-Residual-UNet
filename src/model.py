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

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("VAR_"+var.name.replace(':','_')):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class SGRU(object):

    def __init__(self, inputs):
        self.inputs = inputs

        with tf.variable_scope('SGRU_MODEL'):
            inputs, conv1 = self._swish_gated_block('SGB_1', inputs, 96, conv1x1=False)
            inputs, conv2 = self._swish_gated_block('SGB_2', inputs, 192)
            inputs, conv3 = self._swish_gated_block('SGB_3', inputs, 288)
            inputs, conv4 = self._swish_gated_block('SGB_4', inputs, 384)
            inputs, conv5 = self._swish_gated_block('SGB_5', inputs, 480)

            swish1 = tf.nn.swish(conv1)
            swish2 = tf.nn.swish(conv2)
            swish3 = tf.nn.swish(conv3)
            swish4 = tf.nn.swish(conv4)
            swish5 = tf.nn.swish(conv5)

            inputs, _ = self._swish_gated_block('SGB_5_up', inputs, 512, cat=swish5)
            inputs, _ = self._swish_gated_block('SGB_4_up', inputs, 480, cat=swish4)
            inputs, _ = self._swish_gated_block('SGB_3_up', inputs, 384, cat=swish3)
            inputs, _ = self._swish_gated_block('SGB_2_up', inputs, 288, cat=swish2)
            inputs, _ = self._swish_gated_block('SGB_1_up', inputs, 192, cat=swish1)

            conv1_1_up = Conv2DLReLU(filters=96, kernel_size=1, inputs=inputs)
            conv1_2_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_1_up)
            conv1_3_up = Conv2DLReLU(filters=96, kernel_size=3, inputs=conv1_2_up)
            conv1_4_up = tf.layers.Conv2D(filters=27, kernel_size=1, activation=None,
                    padding='same', kernel_initializer='he_normal')(conv1_3_up)

            self.output = conv1_4_up

        self.params = tf.trainable_variables(scope='SGRU_MODEL')
        with tf.name_scope('summaries'):
            for var in self.params:
                variable_summaries(var)
        self.saver = tf.saver = tf.train.Saver(self.params, max_to_keep=5)

    def _swish_gated_block(self, name, inputs, filters, cat=None, conv1x1=True):
        """swish_gated block takes in a input tensor and returns two objects, one of
        which is the concat operation found in the SGB, and the other is the
        output of the last convolutional layer (before maxpool or deconv)

        (Think of a better variable name than cat)
        If the cat list is an empty list, we assume we are in the down part of the
        Unet. Otherwise, we are in the up part.
        """
        with tf.variable_scope(name):
            if conv1x1:
                inputs = Conv2DLReLU(filters=filters, kernel_size=1, inputs=inputs)

            conv1 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=inputs)
            conv2 = Conv2DLReLU(filters=filters, kernel_size=3, inputs=conv1)

            if cat is None:
                sgb_op = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
                swish = tf.nn.swish(inputs)
                swish = tf.layers.MaxPooling2D(pool_size=2, strides=2)(swish)
                concat = [sgb_op, swish]
            else:
                sgb_op = Conv2DTransposeLReLU(filters=filters, strides=2, inputs=conv2)
                swish = tf.nn.swish(inputs)
                swish = Conv2DTransposeLReLU(filters=filters, strides=2, inputs=swish)
                concat = [sgb_op, swish, cat]

            return tf.concat(concat, axis=3), conv2

    def save(self, path):
        """need to figure out whether path variable needs .ckpt in the name"""
        self.saver.save(tf.get_default_session(), path)

    def load(self, path):
        self.saver.restore(tf.get_default_session(), path)
