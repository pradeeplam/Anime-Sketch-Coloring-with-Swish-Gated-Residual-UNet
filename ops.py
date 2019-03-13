    '''
    Holds all the helper functions
    '''

    def Conv2D_LRELU(filters, kernel_size, padding='same', strides=(1,1), kernel_initializer='he_normal', alpha=0.03, inputs):

        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer) (inputs)
        conv = LeakyReLU(alpha=alpha)(conv)

        return conv

    def Conv2DTranspose_LRELU(filters, kernel_size, padding='same', strides=(1,1), kernel_initializer='he_normal', alpha=0.03, inputs):

        deconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer) (inputs)
        deconv = LeakyReLU(alpha=alpha)(deconv)

        return deconv

    def Swish(cropping, data_format='channels_last', inputs):
        swish = Activation('sigmoid')(inputs)
        swish = Multiply([swish, inputs])
        swish = Cropping2D(cropping=cropping, data_format=data_format)(swish)

        return swish 